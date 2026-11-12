# Experiment log: cheaper multi-hop evidence selection

**Goal:** give the LLM only the evidence it needs for multi-hop QA. Use a selector that is small and cheap (few parameters and FLOPs) instead of sending all retrieved paragraphs.

**Starting point:** BRACIS 2026 Noise-Gate \cite{lavado2027noisegate}: 26.7% fewer tokens at equal F1 on HotpotQA.
**Plan and literature:** [`../research_direction.md`](../research_direction.md)

## Fixed protocol (all experiments)

- **Benchmark:** HotpotQA distractor (10 paragraphs per question: 2 gold, 8 BM25 distractors).
- **Test:** the BRACIS paper's 400 validation questions. **Dev:** 1000 other validation questions, used for tuning. Both are fixed by ID (`exp02_cross_encoder/split_ids.json`).
- **Offline metric (no LLM):** *both-gold* = both gold paragraphs kept; plus prompt tokens on the paper's exact prompt (tiktoken o200k).
- **LLM check:** stateless calls with only the selected paragraphs. Cost comes from counted prompt tokens.
- **Rules:** each experiment lives in its own folder with its own uv environment; thresholds are chosen on dev and reported once on test.

**Reference points (test):**

| Context | Tokens |
|---|---|
| All 10 paragraphs | 1442 |
| Gold paragraphs only | 323 |
| Gold sentences only | 214 |

## Status

| # | Experiment | Folder | Status | Headline |
|---|---|---|---|---|
| exp01 | Hop-aware Noise-Gate (option 1): training-free 2nd-hop threshold on "question + top paragraph" | [`exp01_hop_gate`](exp01_hop_gate/README.md) | ✅ done | 6–10% fewer tokens than Noise-Gate at equal both-gold (0.918 at 1089 tok vs 0.928 at 1221). Precision stays at ~0.25–0.30: thresholds plateau. |
| exp02 | Small learned selector (option 2): 22.7M MiniLM-L6 cross-encoder | [`exp02_cross_encoder`](exp02_cross_encoder/README.md) | ✅ done (offline + gpt-5.4-mini F1) | **F1 (gpt-5.4-mini, 400 q): A1 t=0.02 0.796 at 467 tok vs k=10 0.782 at 1448 (−68% tokens, ΔF1 +0.014, CI includes 0); top-2 0.780 at 330 tok (−77%).** Offline: | **A1 (full fine-tune): 0.915 both-gold at 391 tok (−73% vs k=10, −64% vs hop gate); 0.968 at 461 tok.** A2 n200: 0.915 at 733 tok. A0: 0.915 at 1143 tok. |
| exp03 | LLM weight editing (option 7): Opus 5.5 / Astra edit the 22M cross-encoder with N examples, compared against fine-tuning on the same N | — | ⏳ planned | Go/no-go: linear probe for a "distractor direction" first. |
| — | Sentence-level selection (gold sentences = 214 tok) | — | 💡 idea | — |
| — | Transfer to 2Wiki / MuSiQue without retraining | — | 💡 idea | — |

## exp02 arms (fair comparison for exp03)

| Arm | Selector | Training data | Status |
|---|---|---|---|
| A0 | MiniLM-L6 cross-encoder, off the shelf | none (MS MARCO only) | ✅ 0.915 at 1143 tok |
| A2 | same, fine-tuned | N = 50 / 200 HotpotQA questions | ✅ n50 0.932 at 888 tok; n200 0.915 at 733 tok |
| A1 | same, fine-tuned | all ~90k train questions, 1 epoch (~50 min, T4) | ✅ 0.915 at 391 tok; 0.968 at 461; top-2 0.878 at 324 |
| A3 / A4 | same, **weights edited by an LLM** | N = 50 / 200 | exp03 |

## Findings so far

1. **Multi-hop structure matters, but cosine thresholds plateau.** The hop gate helps, yet only about 1 in 4 kept paragraphs is gold (exp01).
2. **Reading question and paragraph together helps a bit even untrained.** The zero-shot cross-encoder slightly beats the hop gate (exp02 A0).
2b. **A 22.7M fine-tuned selector closes most of the gap.** It keeps both gold in 91.5% of questions at 391 tokens (oracle 323), about 2.6 paragraphs on average. Its estimated cost is ~1/180 of the generator FLOPs it saves (exp02 A1). Small-data fine-tuning (n200) needs ~2× the tokens for the same recall, which leaves room for exp03.
2c. **Answer quality holds.** With the paper's generator, every A1 rule matches k=10 F1 (paired CIs include 0) at 68–77% fewer tokens. Gold-only context *beats* k=10 by +0.040 F1 (CI excludes 0), so distractors hurt the generator.
3. **The literature ceiling is high.** Large trained selectors pick exactly both gold paragraphs 92–97.5% of the time (SAE → Beam Retrieval). Our question is how small and cheap the selector can be.
4. **Embedder mismatch:** the BRACIS Noise-Gate numbers appear to come from OpenAI `text-embedding-3-small` (per the `cart-code` runner), not the MiniLM model stated in the paper. To be checked.

## Log

- **2026-11-09:** research direction and bibliography; exp01 run (hop gate plus 5-question Haiku pilot).
- **2026-11-10:** literature check on learned selectors; exp02 and exp03 designed.
- **2026-11-12:** exp02 pushed to GitHub and run on a Colab T4 (results saved to Drive); A0 zero-shot scored locally and matches Colab. A1 full fine-tune: 0.915 both-gold at 391 tok. gpt-5.4-mini F1 on the 400 questions × 5 contexts (2,000 calls, ~$0.25): A1 matches k=10 at 68–77% fewer tokens.
