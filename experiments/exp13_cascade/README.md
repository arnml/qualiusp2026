# exp13: 4-hop confirmation and a one-shot → iterative cascade

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit d071b74, revised after an adversarial Codex pre-run review).
- **Models:** exp12's frozen GPU models. M1-iter and M1-ctrl were trained on identical pairs.
- **Data:** fresh questions never used before (seed-13 splits).
- **Compute:** Colab T4, code pinned to d071b74. The analysis is pre-written (`analyze.py`).
- **Results:** [`results/results.json`](results/results.json), with per-question hits and router decisions in [`results/hits_compact.json`](results/hits_compact.json). Both were checked locally: the primary recomputed from the hits matches exactly.

## Bottom line

1. **Primary confirmed: on fresh 4-hop chains, conditioning-trained iteration beats matched one-shot ranking.** n = 566, recall@5 **0.500 vs 0.445, +0.055 [+0.007, +0.102]**, 117 vs 86 discordant questions, **p = 0.035**. Both strata are positive, so the success rule is met.
   - **The strata differ a lot:**
     - dev 4-hop (n = 166): **+0.139** (p = 0.006)
     - train 4-hop (n = 400): **+0.020** (n.s.)
   - **Seeds:** +0.055 / +0.044 / +0.016, all positive.
   - **Size:** the effect is at the pre-registered minimum worthwhile effect (+0.05), not the +0.19 seen after the fact in exp12, as the stated prior expected.
2. **The cascade fails both secondaries.**
   - **Router:** routes 29% of questions; 4-hop recall 0.76, precision 0.43.
   - **Cascade:** recall@5 0.702 at 41.5 passes, against one-shot 0.693 at 20 and always-iterative 0.745 at 74.
     - vs one-shot: **+0.009 [−0.010, +0.027]**, p = 0.39 ✗
     - vs iterative: **−0.043 [−0.068, −0.020]**, not non-inferior ✗
   - **Even the oracle cascade loses:** routing exactly the 4-hop questions gives 0.716 at 32 passes. Iteration also helps 3-hop chains on these fresh questions, so a 4-hop-only router leaves gain on the table.
3. **Exploratory (not pre-registered, uncorrected): on all 1,017 fresh dev questions, iteration helps chains of 3 or more hops.**

   | Hops | Iterative | One-shot | Discordant | p |
   |---|---|---|---|---|
   | all | **0.745** | 0.693 | 151 / 98 | 0.0009 |
   | 2 | 0.822 | 0.813 | 51 / 46 | 0.68 |
   | 3 | **0.724** | 0.646 | 56 / 31 | 0.010 |
   | 4 | **0.542** | 0.404 | 44 / 21 | 0.006 |

   This is larger than exp12's overall +0.04 on its 400 test questions (n.s.). Taken together, the evidence points to a real but modest gain from iteration that comes entirely from ≥ 3-hop chains. A pooled analysis would need to be pre-registered.
4. **The router picks questions that look hard, not questions where iteration helps.** It routed 45% of 3-hop questions, yet the cascade's 3-hop recall (0.640) equals one-shot's (0.646). Score shape and question length predict hop count, but not iteration benefit.

## 2×2 on fresh 4-hop chains (recall@5)

| | f4_dev: one-shot | f4_dev: iterative | f4_train: one-shot | f4_train: iterative |
|---|---|---|---|---|
| M1-iter | 0.337 | **0.542** | 0.393 | **0.483** |
| M1-ctrl | **0.404** | 0.169 | **0.463** | 0.118 |

- Each model is best in the mode it was trained for, as in exp12.
- 17–19% of the iterative 4-hop questions reach the 512-token limit at the last step, so the oldest picked paragraph is truncated.

## What this changes

- **H6-4hop:** confirmed (C). The effect is modest (+0.055) and heterogeneous across sources.
- **Cascade (H11):** N. A 4-hop router built from one-shot score features does not make a useful cascade.
- **For the thesis:**
  - *Research:* the conditioning skill (a rank-2 change, exp12) gives a real gain on chains of 3 or more hops, about +0.05 to +0.14 recall@5, and nothing on 2-hop chains.
  - *Practical:* the obvious cascade fails. The open problem is predicting *where iteration helps*, not predicting hop count. A router trained on that benefit directly, labelled by running both rankers on a training split, is the next candidate.

## Deviations

- **The router's cascade behavior was partly seen in a smoke test** before registration: question-length features with random scores gave 4-hop recall 0.75 and precision 0.41. This is disclosed in PROTOCOL, and nothing was changed.
- **Colab reset its runtime once** while waiting for the Drive authorization, before any ranking. The run started from scratch in a fresh output folder.

## Files

- `PROTOCOL.md`, `split_ids.json`, `rank.py`, `analyze.py`, `make_notebook.py`, `exp13_colab.ipynb`
- `results/results.json`, `results/hits_compact.json`
