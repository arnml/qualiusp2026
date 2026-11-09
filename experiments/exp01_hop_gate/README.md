# exp01: hop-aware Noise-Gate (option 1)

Training-free. Adds a second-hop threshold to Noise-Gate (BRACIS 2026) using only sentence embeddings.
Self-contained: own uv environment, reads the HotpotQA snapshot from `cart-code` read-only (override with `HOTPOT_DIR`).

```bash
uv sync
uv run python eval_selection.py          # embeddings (CPU, ~10 min first time, cached) + dev sweep + test
uv run python pilot.py make "hop@0.9" 5  # pick 5 test questions for the LLM pilot
uv run python pilot.py score             # after results/pilot_answers.json exists
```

## Method

- **Hop 1:** keep paragraphs with `cos(q, p) >= tau1`. The top-1 paragraph is always an anchor.
- **Hop 2:** for every other paragraph, `bridge(p) = max over anchors a of cos(p, v_a)`; keep it if `bridge >= tau2`. `v_a` is one of:
  - `concat`: `enc("q a")`, one extra encoder pass per anchor (best variant)
  - `sum`: `normalize(e_q + e_a)`, 0 extra passes
  - `pp`: `e_a`, 0 extra passes
- **Then** the same Jaccard redundancy guard (rho=0.65).

Embedder: `multi-qa-MiniLM-L6-cos-v1` (22M params), as stated in the paper.

## Protocol

- **Test:** the paper's exact 400 HotpotQA-distractor validation questions (`Random(42)`); all 400 IDs match the paper's CSV.
- **Dev:** 1000 *other* validation questions (the train split isn't downloaded). **All thresholds are chosen on dev only.**
- **Offline metric (no LLM):** *full recall* = both gold paragraphs kept. Prompt tokens are counted with tiktoken `o200k_base` on the paper's exact prompt.

## Results

### Equal-recall comparison on dev (tokens needed)

Noise-Gate tokens are read off its tau grid (step 0.025); "≥ target" overshoots, so the savings are approximate.

| Full recall ≥ | Noise-Gate | Hop-gate (concat) | Saving |
|---|---|---|---|
| 0.80 | 1005 (0.816) | 891 (0.807) | ~9% |
| 0.85 | 1072 (0.860) | 957 (0.850) | ~9% |
| 0.90 | 1190 (0.918) | 1069 (0.903) | ~7% |
| 0.95 | 1274 (0.955) | 1191 (0.950) | ~6% |

At equal tokens it is +5 to +7 points of full recall in the 650–1000 token range. Below ~500 tokens, plain top-2 or top-3 beats both methods.

### Test (400), operating points chosen on dev

| Method | Full recall | Precision | Paragraphs | Tokens |
|---|---|---|---|---|
| All 10 (k=10) | 1.000 | 0.20 | 10.0 | 1442 |
| Noise-Gate tau=0.20 | 0.928 | 0.25 | 8.4 | 1221 |
| **Hop-gate concat (0.30 / 0.25)** | **0.918** | 0.29 | 7.6 | **1089** |
| Noise-Gate tau=0.15 | 0.960 | 0.23 | 9.0 | 1309 |
| **Hop-gate concat (0.30 / 0.175)** | **0.958** | 0.25 | 8.5 | **1216** |
| Noise-Gate tau=0.30 (MiniLM) | 0.782 | 0.33 | 6.6 | 951 |
| Oracle: gold paragraphs only | 1.000 | 1.00 | 2.0 | 323 |
| Oracle: gold *sentences* only | — | — | ~2.4 sentences | 214 |

HotpotQA has about 41.5 sentences per question across the 10 paragraphs, of which about 2.4 are supporting facts.

### LLM pilot (5 stateless Haiku subagents, one question each, only the selected paragraphs)

The 5 questions are the shortest of the 54 test questions where the hop gate keeps both gold paragraphs and MiniLM Noise-Gate@0.30 does not. **This is a pipeline sanity check, not a measurement.**

| Question | Gold | Haiku + hop context | F1 | Tokens | Paper k=10 (gpt-5.4-mini) | Paper NG@0.30 |
|---|---|---|---|---|---|---|
| Hush star born 1949 | Jessica Lange | Jessica Lange | 1.0 | 273 | 1.0 / 1185 tok | 1.0 / 153 tok |
| Lugar senate end | 2013 | 2013 | 1.0 | 296 | 1.0 / 907 | 1.0 / 302 |
| Big Stone Gap director city | Greenwich Village, NYC | Greenwich Village | 0.57 | 403 | 1.0 / 1388 | 0.57 / 631 |
| Munsonville population | 729 at the 2010 census | 729 | 0.40 | 405 | 0.40 / 1952 | 0.40 / 812 |
| Brian Mannal city | Boston | Boston | 1.0 | 456 | 1.0 / 1211 | 1.0 / 462 |

Mean F1 is 0.79 at 367 tokens. The paper's k=10 run on the same questions got 0.88 at 1329 tokens. The two partial scores are answer-span-format issues, not missing evidence.

## Findings

1. **Option 1 works, but the gain is small.** It needs about 6–10% fewer tokens than Noise-Gate at equal gold recall, with zero training and one extra MiniLM pass.
2. **The bigger lesson is the headroom.** At ~92% full recall we still send ~1100 tokens, while gold paragraphs need 323 and gold sentences 214. Precision stays at ~0.25–0.30, so threshold rules on cosine cannot separate gold from distractors. This motivates option 2 (a learned gate) and sentence-level selection.
3. **The paper's embedder appears to differ from what the paper states.** `cart-code/experiments/run_noise_gate.py` takes its embedder from `LLM_TO_EMBEDDING["gpt-5.4-mini"]`, which is OpenAI `text-embedding-3-small`. The paper text says MiniLM. The per-question tokens of the paper's NG@0.30 (e.g. 153 tokens on Q1) also differ from MiniLM NG@0.30 here. This needs checking against the paper.
4. **Subagents as the generator:** answers match the paper's gpt-5.4-mini answers on all 5 questions. But each subagent call carries ~54k tokens of harness overhead, so **cost must be computed from prompt tokens (tiktoken), never from subagent usage.**

## Files

- `results/dev_grid.csv`: every config on dev
- `results/test_operating_points.csv`: the test table
- `results/test_selections.json`: per-question selections
- `results/pilot_*.json`, `results/pilot_answers/`: the pilot
