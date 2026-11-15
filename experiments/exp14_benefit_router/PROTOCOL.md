# exp14 protocol: a cascade with a router trained on iteration benefit (pre-registered 2026-11-15)

## Why

exp13 established three things:
- **4-hop gain is real:** conditioning-trained iteration beats matched one-shot ranking on fresh 4-hop chains (+0.055, p = 0.035).
- **The 4-hop cascade failed:** its router predicts *hop count*, not *where iteration helps*.
- **3-hop gains too (exploratory):** on fresh dev questions, iteration also helps 3-hop chains.

**Question:** can a cheap router, trained directly on per-question benefit, capture iteration's gain at a fraction of its cost?

## Models (frozen, exp12 GPU seed 0)

- **One-shot:** M1-ctrl (`m1_ctrl`), 20 passes per question.
- **Iterative:** M1-iter (`m1_iter_s0`), about 74 passes.
- **The cascade:**
  - every question gets the one-shot pass
  - routed questions also get the iterative pass, about 20 + 74
  - routed questions use the iterative ranking
- exp13's rankings of `router_train` and `dev_fresh` were made with these same models (sha recorded) and are reused.

## Data (`split_ids.json`, fixed before any scoring; seed 14)

Drawn from the 13,538 MuSiQue-Ans train questions **never used** in exp08–exp13 (natural hop mix: about 74% 2-hop, 22% 3-hop, 3–4% 4-hop):

| Split | n | Use |
|---|---|---|
| `b_train` | 4,000 | label benefit (run both rankers) and fit the router |
| `b_tune` | 1,000 | choose the routing threshold τ |
| `b_eval` | 2,000 | **all pre-registered tests, once** |
| `router_train` | 1,000 | refit exp13's hop router exactly (comparison only) |
| `dev_fresh` | 1,017 | descriptive only: **re-used** from exp13, where it informed this design |

## Router (fixed now)

- **Label per question:** g = hit@5(iterative) − hit@5(one-shot) ∈ {−1, 0, 1}.
- **Model:** `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)` predicting g. Fit on `b_train`.
- **Features**, all available after the one-shot pass:
  - question word count, commas, " of " count, "'s" count, wh-word count, capitalized words after the first
  - the one-shot model's sorted top-10 scores, the gaps between the top 6, and the softmax entropy of the top 10
- **Threshold τ:** on `b_tune`, from the 101 quantiles of the predicted g. Choose the τ with the **highest cascade recall@5** subject to **mean passes ≤ 50**. Ties go to fewer routed questions.

## Pre-registered tests (b_eval, n = 2,000)

- **Gate:** if always-iterative minus one-shot recall@5 on `b_eval` is **< 0.02**, there is too little benefit to capture, and the primary is reported as **not testable**.
- **Primary:** benefit cascade vs one-shot, recall@5, McNemar exact, two-sided α = 0.05.
  - **Success** = the gate passes, the cascade is higher, p < 0.05, **and** mean passes ≤ 50.
  - **Minimum worthwhile effect:** +0.01 recall@5.
  - **Power:** exp10's train-distribution holdout showed iterative ≠ one-shot on about 19% of questions (unmatched models). If the cascade captures half of a +0.03 gain, about +0.015, then with about 6% discordant questions (120) the test has roughly 50% power. **The CI is the main evidence.**
- **Secondary:**
  - **a.** Benefit router vs exp13's hop router (refit exactly as in exp13), recall@5 McNemar. "Beats" = higher and p < 0.05.
  - **b.** Benefit cascade vs always-iterative: non-inferiority, unrounded 95% CI lower bound > −0.01.
- **Descriptive:**
  - an oracle benefit router (route iff g > 0): the upper bound
  - the AUC of predicted g for "g > 0"
  - routed fraction, passes, recall@k, by hops
  - the frozen router and τ applied to `dev_fresh` (labelled as re-use)

**Claim ledger:** one primary (H12). The secondaries are labelled secondary.

## Compute

- **Colab T4.** About 7,000 questions × (one-shot + iterative), roughly 50–60 min.
- **Inputs:**
  - A1 tokenizer from the Colab Drive (sha-checked)
  - MuSiQue from the official release: dev sha `15fa63…bf3b`; the train subset of 8,000 rows (`b_*` ∪ `router_train`, LF) sha `4093c1…c49b`
  - exp12 models and exp13 rankings from the Colab Drive
- The notebook is pinned to the registration commit and writes to a fresh output folder.

## Honest priors

- **Gate:** may fail. Train-distribution questions are mostly 2-hop, and exp13's train 4-hop stratum gained only +0.02.
- **Benefit is hard to predict:** g is sparse and noisy (about 12% "+1", 7% "−1" in exp10's holdout). An AUC near 0.6 would be typical, and the cascade gain could be within noise.
- **Even a null is informative:** it would say iteration's benefit is not predictable from one-shot signals, so the choice is all-or-nothing.

## Changes after the adversarial pre-run review (Codex, `../reviews/codex_exp14_review.md`; before any ranking)

1. **Primary success** also requires the **point estimate ≥ +0.01**, the minimum worthwhile effect, so a tiny effect cannot pass on p alone. The paired CI is always reported.
2. **The primary is a system comparison** (model + inference). A **2×2 on `b_eval`** (`iter_oneshot`, `ctrl_iter`) is added as descriptive.
3. **Model identity:**
   - the notebook **asserts** both model sha prefixes (`2c91c40f1310ca02`, `ed87244f462412f9`, as in exp13) and prints torch/transformers/GPU versions
   - `analyze.py` asserts the hash, IDs and exact lengths of **every** ranking line it uses, the reused exp13 files included
4. **Secondaries are exploratory** (no multiplicity correction). Non-inferiority uses a **one-sided 95%** paired-bootstrap lower bound > −0.01.
5. **Descriptive additions:**
   - the primary difference split by "iterative query reached 512 tokens"
   - a text-only benefit router (AUC)
   - the bootstrap distribution of τ and routed fraction on `b_tune`
6. **Gate:** if the gate fails, the effect and CI are still reported. A failed gate is not read as "benefit is unlearnable".
7. **Not changed:** passes stay the cost unit (as in exp09–exp13). GPU seconds per configuration are logged by `rank.py`.
