# exp09: iterative (query-updated) reranking on MuSiQue-Ans

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit 51e2616, before training). Test = exp08's 400 MuSiQue-Ans questions; dev300 is used only to pick k for the token comparison. Training: 1,000 MuSiQue train questions; all models start from A1 (MiniLM-L6, 22.7M params), CPU only.

## Bottom line

1. **Primary succeeds.** Re-scoring with "question + paragraphs already picked" beats one-shot scoring trained on the **same** questions.
   - Full recall@5: **0.750 vs 0.682** (M1-iter vs M1); 60 vs 33 discordant questions, McNemar p = **0.0067**.
   - The gain is largest on **4-hop** chains: **0.58 vs 0.36**.
2. **Most of the gain over exp08 comes from in-domain training, not iteration.**
   - Recall@5: A1 0.355 → M1 0.682 → M1-iter 0.750.
   - Tokens at recall 0.90: A1 1,957 → M1 1,533 → M1-iter **1,425**.
3. **Iteration helps the top of the ranking only.**
   - At recall 0.90 it saves **108 more tokens** than M1 [−125, −92] (recall n.s.).
   - At 0.95 it costs 108 more [+92, +125] (14 paragraphs vs 13), so that secondary fails.
   - At k = 10 both reach 0.897. After the 4 greedy picks, the leftover order is no better than one-shot.
4. **Iteration needs training.** Zero-shot A1-iter collapses (recall@5 0.12 vs A1 0.355): A1 never saw long queries. This was the stated prior.
5. **Cost:** 74 cross-encoder passes per question vs 20 (3.7×), plus 5.2 h of CPU training (43 min for M1).

## Numbers (test, 400)

**Full recall@k** (all 2–4 supporting paragraphs in the top k):

| k | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|
| A1 (exp08) | 0.105 | 0.205 | 0.273 | 0.355 | 0.425 | 0.540 | 0.670 | 0.875 |
| A1-iter (zero-shot) | 0.018 | 0.045 | 0.080 | 0.120 | 0.147 | 0.250 | 0.323 | 0.642 |
| M1 (one-shot, MuSiQue) | 0.273 | 0.492 | 0.600 | 0.682 | 0.735 | 0.823 | **0.897** | **0.983** |
| **M1-iter** | **0.310** | **0.530** | **0.672** | **0.750** | **0.785** | **0.850** | **0.897** | 0.970 |

**Recall@5 by hops** (test: 203 / 133 / 64 questions):

| Hops | A1 | M1 | M1-iter |
|---|---|---|---|
| 2 | 0.49 | 0.78 | **0.84** |
| 3 | 0.27 | 0.68 | 0.70 |
| 4 | 0.09 | 0.36 | **0.58** |

**Recall@5 tests** (McNemar exact; "only" = discordant questions):

| Comparison | Role | Recall | Only x / only y | p |
|---|---|---|---|---|
| M1-iter vs M1 | **primary** | 0.750 vs 0.682 | 60 / 33 | **0.0067** |
| M1-iter vs A1 | secondary a | 0.750 vs 0.355 | 173 / 15 | 3e−35 |
| M1 vs A1 | descriptive | 0.682 vs 0.355 | 141 / 10 | 9e−31 |
| A1-iter vs A1 | descriptive | 0.120 vs 0.355 | 10 / 104 | 7e−21 |

**Tokens at a recall target** (secondary b: k chosen on dev300, applied once to test). Δ is a paired bootstrap 95% CI:

| Target | A1 | M1 | M1-iter | M1-iter − M1 | M1-iter − A1 |
|---|---|---|---|---|---|
| 0.90 | k 16: 0.905, 1,957 tok | k 11: 0.920, 1,533 | k 10: 0.898, **1,425** | **−108 [−125, −92]**, recall −0.023 [−0.058, +0.013] ✅ | −532 [−554, −511] ✅ |
| 0.95 | k 18: 0.960, 2,243 | k 13: 0.950, **1,774** | k 14: 0.963, 1,882 | +108 [+92, +125] ✗ | −361 [−378, −344] ✅ |

For scale: sending all paragraphs costs 2,535 tokens. At recall 0.90, M1-iter sends 44% fewer tokens than that, and 27% fewer than A1.

## Reading

- **Why does iteration help at the top?** The step-s query contains the bridge entity found at step s−1. That is exactly the "unnamed bridge" failure from exp05 and the 4-hop collapse from exp08.
- **Why not at 0.95?**
  - Inference uses only 4 greedy steps. Questions whose chain is missed early fall back to step-4 scores, and those rank leftovers worse than M1 does: recall@15 is 0.970 vs 0.983.
  - Two cheap fixes, not tested here: beam search, or merging the one-shot and iterative scores for the tail.
- **Practical comparison with exp07/08:** on HotpotQA, adaptive stopping saved 57–71 tokens. On MuSiQue, a better ranker matters far more:
  - −424 tokens (M1 vs A1) at recall 0.90
  - −532 tokens (M1-iter vs A1)
- **Threats to validity:**
  - one seed, one training size (1,000 questions)
  - M1-iter trains on ≈ 2.2× as many pairs as M1 (18,101 vs 8,338; the same questions). A pair-matched M1 control was not run.
  - test drawn from MuSiQue dev (the official test has no labels)
  - greedy decoding with 4 steps

## Deviations (all recorded before any evaluation)

1. **Budget rule triggered.** The projected time was 12 h, over the 10 h limit, so training used 1,000 questions ([`results/budget.json`](results/budget.json)).
   - The protocol said "the first 1,000". The ids are sorted, so the first 1,000 were all 2-hop.
   - Caught at the start of M1 training (pos = 2000 exactly), before any model was finished. Replaced by a seeded random 1,000 (rng 9; 721 / 220 / 59 two/three/four-hop).
2. **Truncation fallback.** A few paragraphs exceed 512 tokens alone, so `only_first` truncation is impossible. In those batches truncation falls back to `longest_first` (`data.encode`).
3. **Inference max length.** All arms use length 512 (exp08 scored at 256). A1's top 5 matches the exp08 cache on 51 of the first 54 questions; the others differ only in near-ties and truncation.
4. **Machine sleep paused runs twice** (about 8 h and 4 h). Runs resume per question, so no results are affected.

## Files

- `PROTOCOL.md`, `split_ids.json` (`split_ids_make.py`): pre-registration and fixed splits.
- `data.py`, `train.py`, `rank.py`, `analyze.py`, `run_all.sh`: pipeline.
- `results/rank_<arm>_<split>.jsonl`: full rankings.
- `results/results.json`: all tests.
- `results/budget.json`: budget rule and deviation 1.
- `models/` (git-ignored): `m1`, `m1_iter`, with `train_info.json`.
