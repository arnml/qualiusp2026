# exp10: adaptive stopping on the iterative ranker (MuSiQue-Ans)

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit 49248b4, before any scoring).
- **Rankers:** M1 (one-shot) and M1-iter (iterative), frozen from exp09.
- **Stop scorers:** logistic regression, fit on 1,000 unused MuSiQue train questions (holdout).
- **τ:** tuned on dev300.
- **Test:** exp09's 400 questions, used once.
- **Sanity check:** the re-ranked orders equal exp09's on test and dev300 for both rankers (400 / 400, 300 / 300).

## Bottom line

1. **Primary succeeds at recall 0.95, not at 0.90.** Adaptive stopping on M1-iter beats exp09's fixed k:
   - **0.95:** −244 tokens [−296, −192], recall −0.015 [−0.033, +0.003] ✅ (1,638 vs 1,882 tokens)
   - **0.90:** −43 tokens [−95, +9] ✗. The dev300 τ overshot (test recall 0.923 vs 0.898), so the token CI includes 0.
2. **Secondary fails: the iterative score is *not* a better stop signal.** Adaptive M1 (one-shot) uses fewer tokens than adaptive M1-iter:
   - **0.95:** same recall (0.9475 both), but M1-iter costs **+84 tokens** [+39, +129]
   - **0.90:** M1-iter costs +161 tokens, but with higher recall (+0.055). Adaptive M1 undershot (0.868)
3. **The pre-registered mechanism did not appear.** "Is anything left?" did not let M1-iter stop earlier:
   - mean k is 9.5–11.7, higher than M1's
   - M1-iter almost never stops at k ≤ 3, so it still needs **73.5–73.9 passes** per question vs M1's 20
4. **The useful practical result is simpler: adaptive stopping on the cheap one-shot ranker.** At recall 0.95, adaptive M1 needs **1,554 tokens** with 20 passes per question.
   - That is 220 fewer than M1 with a fixed k (1,774, exp09) and 328 fewer than M1-iter with a fixed k (1,882).
   - Those two comparisons are descriptive: they come from different experiments and were not pre-registered tests.

## Numbers (test, 400)

| Policy | Target 0.90: recall / tokens / mean k / passes | Target 0.95: recall / tokens / mean k / passes |
|---|---|---|
| fixed-k M1-iter (exp09) | 0.898 / 1,425 / 10 / 74 | 0.963 / 1,882 / 14 / 74 |
| **adaptive M1-iter** | 0.923 / 1,383 / 9.5 / 73.5 | 0.948 / **1,638** / 11.7 / 73.9 |
| **adaptive M1** | 0.868 / 1,222 / 8.3 / 20 | 0.948 / **1,554** / 11.0 / 20 |

**Pre-registered tests** (paired bootstrap 95% CI; success = token CI entirely below 0, and recall CI upper bound ≥ 0):

| Comparison | Role | 0.90: Δ tokens, Δ recall | 0.95: Δ tokens, Δ recall |
|---|---|---|---|
| adaptive M1-iter − fixed M1-iter | **primary** | −43 [−95, +9], +0.025 [−0.003, +0.053] ✗ | **−244 [−296, −192]**, −0.015 [−0.033, +0.003] ✅ |
| adaptive M1-iter − adaptive M1 | secondary | +161 [+118, +203], +0.055 [+0.018, +0.093] ✗ | +84 [+39, +129], 0.000 [−0.028, +0.028] ✗ |

**Equal-recall frontier** (descriptive; each policy tuned on test itself, so no dev300 noise), in tokens:

| Target | adaptive M1 | adaptive M1-iter |
|---|---|---|
| 0.90 | 1,357 | **1,232** |
| 0.95 | **1,631** | 1,678 |

The same pattern as exp09: iteration helps at moderate recall, where the top of the ranking matters. It does not help at high recall, where the tail matters.

**By hops** (test: 203 / 133 / 64), recall at the 0.95 target:

| Hops | adaptive M1 | adaptive M1-iter | fixed M1-iter |
|---|---|---|---|
| 2 | 0.980 / 1,462 tok | 0.980 / 1,501 | 0.975 / 1,789 |
| 3 | 0.940 / 1,630 | 0.940 / 1,830 | 0.970 / 1,988 |
| 4 | 0.859 / 1,683 | 0.859 / 1,671 | 0.906 / 1,956 |

- **Most of the saving comes from 2-hop questions**, which stop early.
- **4-hop questions stay hard:** both adaptive policies lose about 0.05 recall on them, against fixed k = 14.

## Reading

- **Research:** the hypothesis was that a ranker trained on "remaining gold, given the prefix" produces a stop signal. It was **refuted** in this form.
  - One-shot scores support stopping at least as well.
  - The likely reason: after the greedy steps, M1-iter scores the tail with a long query and ranks leftovers no better than M1 (exp09's recall@15: 0.970 vs 0.983). The stop signal depends on that weak tail.
- **Practical:** adaptivity is the lever, not the ranker type. Adaptive M1 at recall 0.95 is the best cost point so far on MuSiQue: 1,554 tokens and 20 passes, vs 2,535 for sending everything.
- **dev300 tuning is noisy:** τ missed the 0.90 target in both directions (0.923, 0.868). This was the stated prior. A larger tuning set, or tuning on holdout with cross-fitting, would help.
- **Threats to validity:**
  - one seed
  - one stop model (LR on score features)
  - test drawn from MuSiQue dev
  - the comparison with exp09's fixed k uses exp09's dev300 k choice, as pre-registered

## Files

- `PROTOCOL.md`, `split_ids.json`: pre-registration and splits (holdout = unused exp09 train ids)
- `rank_scores.py`, `run_rank.sh`: re-ranking with scores → `results/rank_<arm>_<split>.jsonl`
- `select_stop.py`: stop scorers, τ tuning, tests → `results/results.json`
- Logs: `rank_log.txt`, `analysis_log.txt` (git-ignored)

## Exploratory addendum (2026-11-15): the Adaptive-k gap rule on our scores

Not pre-registered. Asked for by the literature review, which names Adaptive-k as the must-beat baseline.
- **Earlier result:** the BRACIS paper (Noise-Gate) already found Adaptive-k weak on HotpotQA with cosine scores: F1 0.627 vs 0.651 / 0.716 / 0.792 for fixed k = 3 / 5 / 10. The largest gap falls inside the BM25 distractor cluster.
- **Here:** k* = argmax(s_i − s_{i+1}) over the selection scores, on test (400). k* = 1 is counted as k = 2, which slightly overstates recall.

| Policy | Full recall | Tokens | Mean k |
|---|---|---|---|
| Adaptive-k, M1 | 0.503 | 672 | 3.7 |
| Adaptive-k + 5 buffer, M1 | 0.815 | 1,227 | 8.4 |
| Adaptive-k, M1-iter | 0.520 | 532 | 2.9 |
| Adaptive-k + 5 buffer, M1-iter | 0.850 | 1,151 | 7.8 |
| our adaptive M1 (target 0.90) | 0.868 | 1,222 | 8.3 |
| our adaptive M1-iter (target 0.90) | 0.923 | 1,383 | 9.5 |

**Reading:**
- The gap rule also cuts too early on cross-encoder scores.
- With the buffer, it spends the same tokens as adaptive M1 but recovers fewer full chains (0.815 vs 0.868).
- It has no way to aim at a recall target.
- This is consistent with the BRACIS failure and extends it from cosine to cross-encoder scores. It is descriptive only, with no test.
