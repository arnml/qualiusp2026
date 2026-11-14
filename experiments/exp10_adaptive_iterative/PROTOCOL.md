# exp10 protocol: adaptive stopping on the iterative ranker (pre-registered 2026-11-14)

## Question

**Does exp07/08-style adaptive stopping, applied to exp09's iterative ranker (M1-iter), cut tokens below exp09's fixed-k policy at the same recall?** And is the iterative ranker still better than the one-shot ranker (M1) once both stop adaptively?

Why it may work: M1-iter was trained to score "is this one of the *remaining* gold paragraphs, given those already picked?". Once the chain is complete there is no positive left, so the next-step score should drop. That is a direct stop signal, which one-shot scores do not have.

## Data (`split_ids.json`, fixed before any scoring)

| Split | What | Use |
|---|---|---|
| **holdout** | the 1,000 exp09 train ids **not** used to train M1 / M1-iter (732 / 205 / 63 two/three/four-hop) | fit the stop scorers |
| **dev300** | as in exp09 | tune τ |
| **test** | exp09's 400 | the comparisons, once |

## Rankers (frozen, from exp09)

- **M1** (one-shot) and **M1-iter** (iterative). Inference is identical to exp09 (4 greedy steps; the tail is ordered by the last step's scores). Rankings are recomputed to record scores and must equal exp09's orders on test and dev300.
- **Selection score sel_j** for the paragraph at position j:
  - **M1-iter:** the score it received in the scoring that placed it (positions 1–3: the step that picked it; positions ≥ 4: the step-3 scoring, which picked position 4 and ordered the tail)
  - **M1:** the one-shot score

## Stopping scorer and policy (same definition for both rankers)

- **Target:** p_k = P(top k holds all supporting paragraphs), for k = 2 … 20.
- **Model:** logistic regression on:
  - k
  - sel_1 … sel_10, padded with the last value
  - sel_k and sel_{k+1} (the "is anything left?" signal)
  - the gap sel_k − sel_{k+1}
  - min(sel_1 … sel_k)
- **Training:** fit on holdout, one example per (question, k).
- **Policy:** send the top k for the smallest k with p_k ≥ τ (else all paragraphs). τ is tuned on dev300 (fewest mean tokens with recall ≥ target) and applied once to test.
- **Targets:** full recall **0.90** and **0.95**.

## Pre-registered comparisons (test, 400)

The criteria are those of exp07/08: the paired bootstrap 95% CI of the token difference lies entirely below 0, and the recall-difference CI includes 0 or is above it.

1. **Primary:** `adaptive(M1-iter)` vs `fixed-k(M1-iter)`, exp09's policy (k from dev300: 10 at 0.90, 14 at 0.95). **Success** = the criteria hold at either target.
2. **Secondary:** `adaptive(M1-iter)` vs `adaptive(M1)`, same criteria.
3. **Descriptive:**
   - cross-encoder passes per question. M1-iter stopping at k needs scoring steps 0 … min(k, 3), so it saves passes when k ≤ 3; M1 always needs 20.
   - results by hops
   - mean k
   - the equal-recall frontier on test (each policy tuned on test itself)
   - comparison with exp08's A1 numbers

## Honest priors

- **Likely gain over fixed k:** fixed k = 10 sends 10 paragraphs to 2-hop questions that M1-iter often completes by k = 2–3 (recall@2 0.31, @3 0.53).
- **The secondary is uncertain.** exp08 showed one-shot scores support adaptive stopping too. The iterative advantage was at the top of the ranking, which is where adaptive stopping operates, so a gain is plausible.
- **dev300 is small:** τ may be tuned noisily, especially at 0.95.
