# Session handoff (for context compaction): read this first

Last updated: 2026-11-12. Repo: `C:\code\qualiusp2026` (GitHub `arnml/qualiusp2026`, public, branch `main`).
User: Arnold Moya (master's qualification, USP; PhD path later). Previous paper: BRACIS 2026, Noise-Gate (code: `C:\code\cart-code`).

## Goal of the thesis work

Cheaper multi-hop QA: a **small, cheap selector** (few params/FLOPs) decides which retrieved paragraphs the LLM sees, instead of sending all of them. Main benchmark: HotpotQA distractor, the same 400 test questions as the BRACIS paper, plus 1000 dev questions; both are fixed by ID in `experiments/*/split_ids.json`.

## Key files

- `experiments/EXPERIMENTS.md`: **high-level tracker** (status, headline, findings, dated log). Keep it updated.
- `research_direction.md`: options and literature. `bibli.bib` holds all references (arXiv-verified).
- One folder per experiment, each with its own **uv** environment (the user wants each experiment self-contained).

## Results so far

| Exp | Status | Headline |
|---|---|---|
| exp01 hop gate (training-free) | done | 6–10% fewer tokens than Noise-Gate at equal both-gold. Thresholds plateau. |
| exp02 22.7M cross-encoder (`ms-marco-MiniLM-L6-v2`), fine-tuned on a Colab T4 | done | **A1 full fine-tune: gpt-5.4-mini F1 0.796 at 467 tok vs k=10 0.782 at 1448 tok (−68%, paired CI includes 0); top-2 0.780 at 330 tok.** Oracle gold 0.822 (distractors hurt). A2 (n50/n200) and A0 zero-shot are baselines. |
| exp03 LLM weight edit, protocol + probe | probe = **no-go** | The editor-allowed mean-difference direction reproduces A0's score; the best [CLS] linear probe adds only +7 pts top-3, below A2. `PROBE_EXPLAINED.md` explains it simply (the user liked this method; keep it in the dissertation). |
| exp04 Astra edits the selector | **harness built, baselines running, no Astra run yet** | PI override of exp03's no-go (a linear probe can't see non-linear mechanisms). Track C = **compress A1** (remove heads/neurons/layers) vs pruning baselines; Track R = repair A0/A2 (later). Stages: forward-pass budget **B = 5 → 10 → 50**, with gates. |

## exp04 state (`experiments/exp04_astra_edit/`)

- **Committed** (8d1a5c8): `editlib.py` (exact removals, cost), `kernel.py` + `kernel_api.py`, `harness.py`, `prompt_trackC.md` (pre-registered before any run), `baselines.py`, `dummy_script.json`, README, DESIGN.
- **A1 weights:** `models/a1/` (git-ignored; sha256 prefix `605c986eb9a9a82b`). Verified identical to exp02 scores. The original is on Google Drive in `MyDrive/qualiusp2026/exp02_results/full/model`.
- **Dry run** `runs/dry2` passed every check. A1 practice both-gold@3 = 0.945. A full practice pass (1,992 pairs, both models) takes **~5 min on CPU**, so B = 50 means hours of forward time.
- **Finished:** `uv run python baselines.py --eval-n 300` (log `results_baselines_log.txt`, output `results/baselines_dev300.json`). Layer drop, Taylor, Wanda-style and random pruning at FLOPs fractions 0.9–0.5 on 300 dev questions.

**Baselines: DONE.** The full table is in `experiments/exp04_astra_edit/README.md`. At ~60% FLOPs Wanda reaches 0.917; at ~50%, 0.820; Taylor reaches 0.823 and 0.720; random 0.797 and 0.443. Next: `evaluate.py`, then stage 1.

*(Earlier partial snapshot: 14 of 21 configs; dev both-gold@3, 300 questions; A1 unedited = 0.970 at 39.26 GFLOPs; the target fraction is approximate, actual GFLOPs are in the json):

| Method | ~90% FLOPs | ~80% | ~70% | ~60% | ~50% |
|---|---|---|---|---|---|
| Layer drop | 0.767 (−layer 5) | 0.290 (−layers 4–5) | 0.290 (same) | 0.020 (−3 layers) | pending |
| Taylor | 0.943 | 0.927 | 0.877 | pending | pending |
| **Wanda-style** | **0.970** | **0.950** | **0.930** | pending | pending |
| Random | 0.953 | 0.913 | 0.887 | pending | pending |

- Wanda is the strongest baseline. It loses 0, 2 and 4 points at roughly 10%, 20% and 30% FLOPs saved.
- Layer dropping collapses.
- Taylor is no better than random here, probably because gradient importance on the 200 practice questions is noisy.
- **Astra's bar:** beat Wanda at equal FLOPs, most interestingly at 60–50%.

### Next steps

1. Read the baseline results; add them to the exp04 README and EXPERIMENTS.md; commit + push.
2. **Stage 1:**
   ```bash
   uv run python harness.py --run s1_C_B5 --track C --budget 5 --turn-cap 40 --editor astra
   ```
   (cwd `experiments/exp04_astra_edit`). Run it in the background. Each turn is one `codex exec` call (~20k tokens of overhead, ChatGPT-plan limits).
3. **Evaluate the final edit:** we still need an `evaluate.py` that builds `el.build_model(models/a1, EditState.load(run/work/edited))`, scores dev (and at stage 3, test), and reports both-gold@3, tokens at the 0.90 threshold, `cost()` and CPU latency. Compare with the baseline curve at the same FLOPs.
4. **Gate:** go to stage 2 (B = 10, turn cap 80) only if stage 1 is valid and shows a change; go to stage 3 (B = 50, turn cap 200) only if Astra ≥ the best baseline at equal FLOPs. Stage 3 adds test + gpt-5.4-mini F1 (`exp02_cross_encoder/generate.py` pattern).

## Operational gotchas

- **GitHub:** push as **arnml**. The active `gh` account sometimes switches to `amoyavt` (push fails with 403). Fix: `gh auth switch --hostname github.com --user arnml`. The user wants everything committed and pushed ("keep all tracked").
- **OpenAI key:** the session env var may be stale. Read it fresh in PowerShell:
  ```powershell
  $env:OPENAI_API_KEY=[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
  ```
  Never print the key. Generator = `gpt-5.4-mini`. Retry/backoff is in `generate.py`; the user worries about rate limits.
- **Codex/Astra:** model `gpt-6-astra` is available in the user's Codex (ChatGPT login). The tool-off flags are in `harness.py` (`--disable` the feature list, `-c mcp_servers.node_repl.enabled=false`, `-c mcp_servers.MCP_DOCKER.enabled=false`, unquoted `-c web_search=disabled`). `resume` must repeat `-m gpt-6-astra`.
- **transformers 5.22 ignores `head_mask`.** Use hooks or editlib for head removal.
- **Colab:** the user's free tier has a T4. Downloads from Drive's web UI don't save from the built-in browser; use a Colab cell with `files.download()` (the user approves the Drive mount popup). Avoid leaving extra Colab sessions open ("too many sessions" error).
- **Windows / Git Bash + PowerShell.** Python is only available via `uv run`. Use `PYTHONIOENCODING=utf-8`.
- **Subagents as LLM generators:** only for tiny pilots (~54k tokens of overhead each). Haiku subagents must write their answer to a file; the hand-back works.

## User preferences

- Straightforward but innovative; honest science: report negative results, pre-register before running, record deviations.
- Explain things simply when asked. Write plain-language md files for methods worth keeping.
- Each experiment self-contained (own dir + uv env). Save results durably (Drive + repo).
- Go step by step with gates (5 → 10 → 50); avoid big bursts of agent or API calls.

## Update 2026-11-12, evening: exp04 stage 1 DONE

- **Run** `experiments/exp04_astra_edit/runs/s1_C_B5`: Astra, Track C, B = 5, 11 turns, 0 tool violations.
- **Final edit:** 3,584 MLP neurons removed (384/512/512/512/640/1,024 in layers 0–5), no heads or layers. 29.90 GFLOPs (−23.8%).
- **Held-out:** dev300 0.963 (A1 0.970; Wanda 0.950). **dev1000: A1 0.959, Astra 0.946, Wanda 0.945 (tie), Taylor 0.923, random 0.895** (`results/baselines_dev1000_matched_s1.json`). Paired McNemar: 25 vs 24 discordant questions, p = 1.0 (a true tie).
- **Harness deviations:** the `/work` path (fixed in the prompt for stage 2+). Astra also wrote to `C:\work\`; the files are copied into the run folder, and `C:\work` is left for the PI to delete.
- `baselines.py` overwrites `results/baselines_dev{N}.json`. For extra fractions, rename the output afterwards (the matched s1 rows are saved as `baselines_dev300_matched_s1.json`).
- **Next:** finish the dev1000 comparison → apply the stage gate → stage 2:
  ```bash
  uv run python harness.py --run s2_C_B10 --track C --budget 10 --turn-cap 80 --editor astra
  ```

## Update 2026-11-12, night: exp04 stage 2 DONE and evaluated

- **Run** `runs/s2_C_B10`: Astra, B = 10, 15 turns, 0 violations. Frozen edit mlp_C: 4,096 neurons (512/512/512/640/640/1,280), 28.57 GF (−27.2%), 7.50M params.
- **dev1000:** Astra 0.940 = Wanda 0.940 at the same FLOPs. Taylor 0.897, random 0.880, A1 0.959. McNemar 31 vs 31, p = 1.0.
- **Wanda curve (dev1000):**

  | FLOPs kept | 60% | 65% | 70% | 72.8% | 80% | 85% |
  |---|---|---|---|---|---|---|
  | both-gold@3 | 0.909 | 0.916 | 0.930 | 0.940 | 0.953 | 0.959 |

  Wanda is lossless at 85%.
- **Primary criterion (cost at equal quality):** a tie in FLOPs at both stages. Params: Astra −3.1% vs Wanda (stage 2), −2.1% (stage 1).
- **Next:** the PI decides whether to run stage 3 (B = 50, turn cap 200), stop Track C, or pivot (e.g. Track R, or ask Astra to aim for the lossless point). Pending: delete `C:\work` (PI).

## Queued by the PI (2026-11-12, during stage 3): Astra edits the NON-fine-tuned encoder

Run this **after stage 3 is finished and evaluated**:
- **Start model:** A0 = off-the-shelf `cross-encoder/ms-marco-MiniLM-L6-v2` (the model A1 was fine-tuned from), edited directly by Astra.
- **This is Track R** in DESIGN.md: repair A0 for HotpotQA by weight editing, compared with A2 (fine-tune on the same N practice questions) and with A1.
- **Prior expectation:** exp03 probe = no-go (A0 has no linear [CLS] direction to unlock). Keep this honest in the write-up; the run tests non-linear routes the probe can't see.
- **Possible second arm:** Track C on A0 (compress A0 and compare with Wanda on A0).
- **Needs:**
  - a Track R prompt (raw edits, 1% edit-size cap, already in editlib)
  - a pre-registration commit
  - stages 5 → 10 → 50 as before

## Update 2026-11-12, late night: exp04 stage 3 DONE; Track C closed

- **Run** `runs/s3_C_B50`: B = 50, lossless prompt, 23 turns, 0 violations. Edit: 2,016 neurons (96 × layers 0–4 + the whole layer-5 MLP), 34.00 GF (−13.4%), 9.10M params.
- **dev1000:** 0.957 (A1 0.959; Wanda at the same FLOPs 0.962; McNemar 9 vs 14, p = 0.40). **test400:** 0.9375 (A1 0.940, Wanda 0.9425). **CPU:** 219 ms vs A1 241 vs Wanda 217.
- **Verdict:** lossless, but a TIE with Wanda (~1.2 GF gap, under the 2 GF margin). `final_eval.py` does physical pruning plus latency.
- **gpt-5.4-mini F1 on stage 3:** not run yet (pre-registered; ~1,200 calls). Low value given the tie; ask the PI.
- **Next:** the queued A0 / Track R experiment (see above). `C:\work` still needs deleting by the PI (blocked for Claude).

## Update 2026-11-13: Track R stage R1 DONE (no repair)

- `runs/r1_R_B5`: Astra on A0, B = 5, 16 turns, 0 violations. It submitted an **empty edit** (its layer-4 candidate failed its own confirmation). Pre-registered verdict: no repair; the R1 gate is not met.
- Exploratory: the rejected candidate on dev1000 has no effect (1,097 vs 1,089 tok@90). `evaluate_r.py` works (`--refs`, `--run`, `--test`).
- **Decision pending (PI):** stop Track R as pre-registered, or run R2 (B = 10) as a recorded deviation.

## Update 2026-11-13: exp04 CLOSED (PI: "stop and write up")

- Write-up: `experiments/exp04_astra_edit/FINDINGS.md` (question, method, Track C and R results, claims, limitations, run index).
- Track R stopped after R1, as pre-registered. Not run: R2/R3, and gpt-5.4-mini F1 for stage 3 (judged low value given the tie).
- Still open for the PI: delete `C:\work`. Thesis next steps: options in `research_direction.md` (e.g. CPU latency story, sentence-level selection, transfer to 2Wiki/MuSiQue).

## Update 2026-11-13, evening: exp05, exp06, exp07 and the Opus replication done

| Experiment | Result | Where |
|---|---|---|
| exp04, second editor (Opus 5.5) | Compression: −6.6% FLOPs, ties Wanda at equal FLOPs (p = 0.375), loses on cost by 2.3–3.4 GF. A0 repair: no edit (null replicates). 0 violations in both | `exp04_astra_edit/FINDINGS.md` |
| exp06 skill size | A1 − A0 is **low-rank, not sparse**: rank 4 recovers 80% of the tok@90 gain; Astra's 108k-entry cap 3.5%. The skill sits in L3–L4 | `exp06_skill_size/README.md` |
| exp05 failure anatomy | A1 misses = unnamed bridge paragraphs, half at rank 4; the title-link feature does not separate them | `exp05_failure_anatomy/README.md` |
| exp07 adaptive selector | **Primary succeeds:** −57/−71 tokens at recall 0.90/0.95 on test. F1 n.s. (−0.006 at 0.95). Pair-vs-top-2 n.s. | `exp07_chain_selector/README.md` |

**Open for the PI:**
1. Are gradients allowed? This unblocks "Astra repair v2" with a rank budget in L3–L4.
2. Is the thesis contribution exp07 (the practical, cheap adaptive selector) to be tested on MuSiQue/2Wiki next?

## Update 2026-11-13, night: exp08 (exp07 on MuSiQue) done

| Result | Where |
|---|---|
| Pre-registered primary/secondary **not testable** (A1 top-10 full recall 0.68 < targets); F1 not run | `experiments/exp08_musique/README.md` |
| Exploratory k ≤ 20: adaptive_full −62 / −100 tok vs A1 threshold at 0.90 / 0.95 (recall n.s.); text = a1only; A0 top-10 0.51 | `results/results_kmax20.json`, `results/a0_vs_a1_topk_test.json` |

**Open for the PI:** next direction: iterative reranking on MuSiQue (a), MuSiQue-fine-tuned A1 (b), or Astra repair v2 (needs the gradients decision).
