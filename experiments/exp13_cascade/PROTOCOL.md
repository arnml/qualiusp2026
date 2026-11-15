# exp13 protocol: 4-hop confirmation and a one-shot → iterative cascade (pre-registered 2026-11-15)

## Why

exp12 (matched training pairs) found:
- **Overall:** iteration adds only +0.04 recall@5 (n.s.).
- **4-hop chains (exploratory):** 0.578 vs 0.391 (p = 0.036, n = 64), and almost nothing on 2–3 hops.

An effect seen exploratorily must be re-tested on fresh data before anyone claims it (`HYPOTHESES.md`). If it holds, the practical design is a **cascade**: cheap one-shot ranking by default, and the conditioning-trained ranker only where it helps.

## Models (frozen, from exp12; Colab Drive `exp12_975371a/models`)

- **M1-iter** = `m1_iter_s0` (iterative inference). **M1-ctrl** = `m1_ctrl` (one-shot). Both were trained on identical pairs.
- **Seeds 1–2** (`*_s1`, `*_s2`) are run for the primary only (descriptive).
- Inference is exp09/exp12's. Passes = cross-encoder forward passes per question.

## Data (`split_ids.json`, fixed before any scoring; seed 13)

| Split | n | What | Use |
|---|---|---|---|
| `f4_dev` | 166 | all **unused** MuSiQue-Ans dev 4-hop questions (not in exp08/exp09 splits) | primary, stratum 1 |
| `f4_train` | 400 | seeded sample of the 789 unused MuSiQue-Ans train 4-hop questions | primary, stratum 2 |
| `router_train` | 1,000 | exp10's holdout (unused exp09 train ids, natural hop mix) | fit the router only |
| `dev_fresh` | 1,017 | all unused dev questions (529 / 322 / 166 two/three/four-hop) | cascade evaluation |

No question here has been used to train or select anything before. `f4_dev` ⊂ `dev_fresh` by construction. `f4_train` and `router_train` are disjoint. Training-split questions are unseen by the models, which trained on exp09's 1,000 `train` ids only.

## Pre-registered tests

1. **Primary (H6-4hop):** on the 566 fresh 4-hop questions (`f4_dev` ∪ `f4_train`), recall@5 of M1-iter (iterative) vs M1-ctrl (one-shot), McNemar exact, two-sided α = 0.05.
   - **Success** = M1-iter higher and p < 0.05.
   - **Also reported:**
     - each stratum separately ("both positive" is a consistency check, not a test)
     - a paired-bootstrap CI of the difference
     - the seed 1/2 differences
   - **Power:** exp12's 4-hop discordance rate was 0.44, so about 250 discordant questions are expected. The minimum detectable effect at 80% power is about **+0.08** (exp12 observed +0.19, which is likely inflated because it was selected after the fact).
2. **Router (fixed before seeing any exp13 result):** logistic regression for "is 4-hop".
   - Fit on `router_train`.
   - **Features:** question length in words, comma count, and the sorted top-10 scores of the one-shot model (M1-ctrl). All are available after the one-shot pass, at no extra cost.
   - **Class-balanced weights** (`class_weight="balanced"`), because only 6.3% of `router_train` is 4-hop vs 16% of dev. **Threshold 0.5, with no tuning.**
3. **Secondary (cascade, `dev_fresh`):** the cascade = M1-ctrl one-shot for every question, then M1-iter iterative for routed questions. A routed question pays both: about 20 + 74 passes.
   - **a. Cascade vs one-shot:** recall@5, McNemar. "Beats" = higher and p < 0.05.
   - **b. Cascade vs always-iterative:** non-inferiority. The lower bound of the 95% paired-bootstrap CI of (cascade − iterative) must be **> −0.01**.
   - **Mean passes are reported for all policies.**
4. **Descriptive:**
   - an **oracle cascade** (route exactly the 4-hop questions), as an upper bound for routing
   - the router's 4-hop recall and precision
   - recall@k for k = 2 … 15, by hops

**Claim ledger:** the primary is confirmatory for "H6-4hop". Secondaries 3a and 3b are confirmatory for the cascade. Error rates are per claim.

## Analysis

`analyze.py`, written before any exp13 ranking. Bootstrap: 5,000 resamples, seed 0.

## Compute

- **Colab T4** (`exp13_colab.ipynb` from `make_notebook.py`, pinned to the registration commit). Rankings take minutes.
- **Inputs:**
  - A1 tokenizer from the Colab Drive (`exp02_results/full/model`, sha-checked)
  - MuSiQue from the official release (dev sha `15fa63…bf3b`; the train subset of the 1,400 `router_train` ∪ `f4_train` rows, LF, sha `c47ce5…65fe`)
  - exp12 models from the Colab Drive

## Honest priors

- **Primary:** likely positive, but the true 4-hop effect is probably smaller than +0.19, since that was the most favorable stratum chosen after the fact. A null is informative: it would mean exp12's 4-hop effect was noise.
- **Cascade vs one-shot:** probably a small gain overall, because only about 16% of questions are 4-hop.
- **Cascade vs iterative:** non-inferiority is plausible, since iteration helps little on 2–3 hops. Routing errors may break it.
- **Cost:** routing everything costs more than always-iterative (20 + 74), so the cascade saves passes only when the router routes fewer than about 75% of questions.

## Changes after the adversarial pre-run review (Codex, `../reviews/codex_exp13_review.md`; before any ranking)

1. **One primary claim.**
   - **Primary success** = pooled McNemar p < 0.05, M1-iter higher, **and both strata (f4_dev, f4_train) positive.** This is the consistency rule for calling it a general 4-hop effect.
   - **Minimum worthwhile effect:** +0.05 recall@5 on 4-hop (about 28 questions of 566). Power for that effect is only about 45%, so **the paired CI is the main evidence** for its size, and a null result is reported as "below the detectable effect", not "no effect".
2. **The cascade claims are secondary,** each at its own α. Secondary 3b uses the **unrounded** two-sided 95% CI lower bound. The margin of 0.01 (one recall point, about 10 of 1,017 questions) is conservative, set below exp12's whole overall gain.
   - **"Useful cascade"** = 3b holds **and** mean passes ≤ 50 (always-iterative is about 74, one-shot 20).
3. **Sensitivity check:** cascade recall is also reported on `dev_fresh` without the 166 `f4_dev` questions (n = 851), since these overlap the primary.
4. **2×2 on 4-hop (descriptive):** `iter_oneshot` and `ctrl_iter` are added on f4_dev and f4_train. The primary is a **system** comparison; the 2×2 separates model from inference.
5. **Integrity checks:**
   - every ranking line stores the model's sha256 and the device
   - a configuration is skipped only when the same hash is already on file
   - GPU is asserted; orders are full permutations; there is exactly one line per configuration
   - the fraction of iterative queries reaching 512 tokens (truncation) is reported
6. **Provenance of router_train:** exp10 used these 1,000 ids only to fit its stopping scorers (logistic regression on score features). They were never used to train or select a ranker, or to choose features here.

## Disclosure (before registration)

The local smoke test of `analyze.py` used **random** rankings and scores, but the **real** question features (word count, commas) and real hop labels on `dev_fresh`. It therefore showed, incidentally, how a router based on question length routes there: 4-hop recall 0.75, precision 0.41, with random score features. No design choice was changed after seeing this, and the router specification above was fixed before it.
