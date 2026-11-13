# exp08: the exp07 adaptive selector on MuSiQue-Ans

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit 0d1aa07, before any scoring). Test = 400 questions from MuSiQue-Ans dev, dev = 1,000 (disjoint), pool = 4,000 from train. A1 = the HotpotQA cross-encoder, frozen.

## Bottom line

1. **Pre-registered outcome: not testable, so not a success.** A1 transfers poorly. Its top 10 contains all supporting paragraphs for only **0.68** of dev questions. The pre-registered policy stops at k = 10, so it cannot reach recall 0.90 or 0.95 on dev, and no τ exists. The primary and secondary tests both fail. Under the protocol, **the F1 follow-up was not run.**
2. **Exploratory (k up to 20, i.e. the policy can send every paragraph):** `adaptive_full` beats the A1 threshold.
   - At recall 0.90: **−62 tokens** [−97, −27]; recall +0.015 [−0.010, +0.040].
   - At recall 0.95: **−100 tokens** [−131, −68]; recall −0.013 [−0.035, +0.007].
   - That is about 4–5% of a ~1,700–2,000-token prompt. The whole gain comes from A1's pooled vectors: `text` (title/link features) does not beat `a1only`.
3. **Practical reading:** on MuSiQue, single-shot ranking by a 22M cross-encoder needs **14–16 of ~20 paragraphs** to reach recall 0.90–0.95. That saves only 20–32% of tokens vs sending everything (2,535 tokens). The bottleneck is the ranker, not the stopping rule.

## Numbers

**Top-k full recall** (all supporting paragraphs in the top k), test:

| k | 2 | 4 | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| A1 (HotpotQA fine-tuned) | 0.10 | 0.27 | 0.41 | 0.53 | **0.67** | 0.74 | 0.88 | 1.0 |
| A0 (off-the-shelf, exploratory) | 0.11 | 0.26 | 0.36 | 0.43 | 0.51 | 0.57 | 0.66 | 1.0 |

The HotpotQA fine-tune helps (+0.16 at k = 10), but neither model is close to HotpotQA's numbers (A1 kept both gold paragraphs in its top 3 for 0.95 of HotpotQA questions). Dev and pool curves are similar: dev top-10 0.68, pool 0.72.

**Pre-registered run (k ≤ 10), test** — [`results/results.json`](results/results.json):

| Target | A1 threshold (t tuned on dev) | adaptive (all three scorers) |
|---|---|---|
| 0.90 | recall 0.877, 1,717 tok, 14.0 paragraphs | unreachable on dev |
| 0.95 | recall 0.958, 2,017 tok, 16.2 paragraphs | unreachable on dev |

**Exploratory run (k ≤ 20), test** — [`results/results_kmax20.json`](results/results_kmax20.json). Same code, `--kmax 20`, τ tuned on dev:

| Target | Policy | Recall | Tokens | Paragraphs | Δtok vs A1 thr [95% CI] | Δrecall [95% CI] |
|---|---|---|---|---|---|---|
| 0.90 | A1 threshold | 0.877 | 1,717 | 14.0 | | |
| | adaptive_a1only | 0.892 | 1,793 | 14.6 | +76 [+47, +106] | +0.015 [−0.010, +0.040] |
| | adaptive_text | 0.912 | 1,781 | 14.6 | +63 [+30, +96] | +0.035 [+0.013, +0.060] |
| | **adaptive_full** | 0.892 | **1,656** | 13.4 | **−62 [−97, −27]** | +0.015 [−0.010, +0.040] |
| 0.95 | A1 threshold | 0.958 | 2,017 | 16.2 | | |
| | adaptive_a1only | 0.953 | 2,110 | 17.0 | +94 [+66, +121] | −0.005 [−0.020, +0.010] |
| | adaptive_text | 0.953 | 2,117 | 17.1 | +101 [+67, +131] | −0.005 [−0.022, +0.013] |
| | **adaptive_full** | 0.945 | **1,917** | 15.3 | **−100 [−131, −68]** | −0.013 [−0.035, +0.007] |

- **Chain value (exploratory secondary):**
  - `full` vs `a1only`: −138 [−172, −102] tokens at 0.90 and −194 [−224, −161] at 0.95, recall n.s.
  - `text` vs `a1only`: n.s. at both targets (−12 [−32, +8]; +7 [−12, +25]).
  - So the useful signal is in A1's internal vectors, not in title/link features. This matches exp05: title links do not separate failures.
- **Equal-recall frontier (each method tuned on test itself, descriptive):**

  | Target | A1 threshold | a1only | text | full |
  |---|---|---|---|---|
  | 0.90 | 1,785 | 1,809 | 1,757 | **1,676** |
  | 0.95 | 1,970 | 2,032 | 2,111 | **1,955** |

  At 0.95 the frontier gap is only 15 tokens. Part of the dev-tuned −100 comes from `full` landing at slightly lower recall (0.945 vs 0.958). Read the 0.95 gain as small.
- **By hops** (recall at the 0.90 target, A1 threshold → full):
  - 2-hop: 0.96 → 0.95, at 1,689 → 1,512 tokens
  - 3-hop: 0.85 → 0.87
  - 4-hop: 0.67 → 0.77

  4-hop questions are where any single-shot ranker fails.

## Honest reading against the priors

- **"A1 transfer is uncertain"** — confirmed, more strongly than expected. Recall at small k is very low (top-4 0.27), so the adaptive range chosen from HotpotQA (2–10) was wrong for MuSiQue. This is a pre-registration design error, reported as such; the exploratory k ≤ 20 run is the fix and is labelled exploratory everywhere.
- **"More room than on HotpotQA"** — no. The room exists only if the ranker puts the gold paragraphs near the top, and it does not.
- **"Chain features may matter more"** — not as title/link features (`text` = `a1only`). The pooled-vector MLP helps, mainly on 2-hop questions (−177 tokens) and in 4-hop recall.
- **Implication for the thesis:** an adaptive stopping rule is a cheap add-on that keeps helping (HotpotQA −57/−71; MuSiQue exploratory −62/−100). But on MuSiQue-style questions one-shot reranking is the limit. The next lever is **iterative / query-updated retrieval** (re-score with the question + already-selected paragraphs), not a better stopping rule.

## Deviations and fixes (all after seeing results; none change the pre-registered outcome)

- `select_adaptive.py` crashed on the unreachable target (`tune_adaptive` returned None). It now records `unreachable` / `not_testable`, which counts as failure.
- Added `--kmax` (default 10 = pre-registered; 20 = exploratory, writing `*_kmax20.json`). With k > 10 the `a1only` features use the top-(k_max + 1) scores instead of the top-10. Some questions have fewer than 20 paragraphs (minimum 16), so k is clamped for the gap feature.
- `a0_topk.py` (exploratory): A0 vs A1 top-k recall and the all-paragraphs token count on test.
- F1 follow-up **not run** (protocol: only if the primary succeeds). An exploratory F1 for k ≤ 20 would cost about $0.40 if wanted.

## Files

- `PROTOCOL.md`: pre-registration.
- `split_ids.json`: fixed splits.
- `data.py`, `score.py`: A1 cache (git-ignored `cache/`; about 63 min CPU for 5,400 questions).
- `select_adaptive.py`: scorers, policies, tests.
- `results/results.json`, `results/test_selections.json`: pre-registered.
- `results/results_kmax20.json`, `results/test_selections_kmax20.json`: exploratory.
- `results/a0_vs_a1_topk_test.json`.
- `generate_f1.py`: prepared, not run.
