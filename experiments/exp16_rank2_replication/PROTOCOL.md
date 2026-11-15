# exp16 protocol: does the rank-2 conditioning change replicate on other multi-hop datasets? (pre-registered 2026-11-15)

## Why

The central research claim rests on one dataset (MuSiQue). On MuSiQue (exp12), the change that lets a small reranker use a query conditioned on the paragraphs it already picked (D = M1-iter − M1-ctrl, pair-matched) is functionally rank 2:
- the rank-2 graft recovers 86.8% [77.7, 95.8] of the gap
- a random rank-2 change recovers 7%

**This experiment replicates the same design on two more multi-hop datasets.** If the result holds on both, the claim covers three datasets.

## Datasets (pinned Hugging Face revisions; sha256 checked in the notebook)

| Dataset | Source | Paragraphs | Supporting paragraphs | Evaluation splits (`split_ids.json`, seed 16) |
|---|---|---|---|---|
| **2WikiMultiHopQA** (`tw`) \cite{ho2020twowiki} | `xanhho/2WikiMultihopQA` @612bc50 (mirror of the official release); dev sha `c0d8b6…01d52` | 10 | 2 or 4 (comparison, compositional, inference, bridge-comparison) | `tw_test` 1,000 and `tw_tune` 300 from dev |
| **HotpotQA** distractor (`hp`) | `hotpotqa/hotpot_qa` @1908d6a; validation sha `c20b63…72f7c6` | 10 | 2 (bridge, comparison) | `hp_test` 1,000 and `hp_tune` 300 from validation, **excluding all 1,400 ids used in exp01–exp07** |

- **Training questions:** for each dataset, 1,000 questions sampled from its train split (sorted ids, `numpy` rng seed 16; `data.load`).
- **Chain order:** the first appearance of each supporting title in `supporting_facts`. On 2Wiki dev this matches the order of the evidence triples for 95% of compositional questions; comparison questions have no natural order.
- **Caveat:** A1 was fine-tuned on the whole HotpotQA train split (one-shot), so the HotpotQA training questions are not new to A1. Both models of the pair start from A1, so the comparison stays matched.

## Models (Colab T4, seed 0)

Per dataset, from A1, exactly as exp12:
- **`{ds}_iter`:** one training example per reasoning step; the query is the question plus the supporting paragraphs of the previous steps, and the positives are the remaining supporting paragraphs.
- **`{ds}_ctrl`:** identical (paragraph, label) pairs in the same order; the query is the question only.

**Shared settings:** 6 random negatives per step (fewer if not enough non-gold paragraphs), BCE, AdamW lr 2e-5, batch 16, 1 epoch, max length 512.

## Inference and metric

- **Inference:** one-shot, or iterative with 4 greedy steps (exp09–exp13); the tail is ordered by the last step.
- **Primary metric: full-support recall@(|gold| + 1).** It counts a question only if *all* its supporting paragraphs are in the top (number of supporting + 1).
  - With 10 paragraphs, a fixed k = 5 would be close to the ceiling for 2-hop questions. One paragraph of slack is the analogue of MuSiQue's top 5 of about 20.
  - Recall@5 is also reported.

## Grafts (no data used to choose them)

- **Delta:** D = `{ds}_iter` − `{ds}_ctrl`, grafted onto `{ds}_ctrl` and run iteratively.
- **Low-rank grafts:** rank-r truncated SVD of each 2-D delta, 1-D deltas in full, for r ∈ {1, 2, 4, 8}.
- **Random control:** a norm-matched random rank-2 change.
- **Recovery** = (R(graft) − R(`ctrl_iter`)) / (R(`iter_iter`) − R(`ctrl_iter`)).

## Pre-registered test, per dataset

The **rank is fixed at 2 in advance**, which is the claim being replicated; there is no rank selection on the tuning set.

1. **Gate (tuning set):** R(`iter_iter`) − R(`ctrl_iter`) ≥ 0.05. Otherwise the dataset is reported as not testable.
2. **Control (tuning set):** the random rank-2 graft recovers less than half of what the real rank-2 graft recovers.
3. **Test set:**
   - the rank-2 graft's recovery point estimate is ≥ **0.80**, and
   - the lower bound of its 95% paired-bootstrap CI (5,000 joint resamples of questions; a resample with denominator ≤ 0 counts as 0) is > **0.5**.
- **"Holds"** = 1, 2 and 3 are all true.
- **Replication claim:** "rank 2 replicates" if it holds on **both** datasets (three datasets with MuSiQue), and "partial" if it holds on one.

**Descriptive:**
- the smallest rank reaching 80% on the tuning set
- recovery at the other ranks on test
- the 2×2 (model × inference)
- the iteration effect (`iter_iter` vs `ctrl_oneshot`, McNemar) overall and by question type
- recall@5

**Error rates:** two confirmatory tests, one per dataset, each at its own rule (no correction).

## Compute

- **Colab T4:** 4 training runs (about 5–10 min each), then evaluations (minutes). Roughly 1 h in total.
- **Inputs:**
  - A1 from the Colab Drive (sha-checked)
  - datasets downloaded from the pinned revisions
- The notebook is pinned to the registration commit and writes to a fresh output folder.

## Honest priors

- **Gate:** likely passes. A control model never trained on conditioned queries collapses when run iteratively, as on MuSiQue (0.41).
- **Rank 2:** plausible. The same architecture and the same kind of change are involved, but the 10-paragraph, mostly 2-hop setting is easier, and the change may be even simpler (rank 1) or noisier.
- **Iteration effect:** likely small on these mostly 2-hop datasets (exp13: about 0 on 2-hop MuSiQue questions), possibly positive on 2Wiki's 4-paragraph bridge-comparison questions. It is not the confirmatory question here.

## Changes after the adversarial pre-run review (Codex, `../reviews/codex_exp16_review.md`; before any training)

1. **Seed-noise null:** a second control per dataset (`{ds}_ctrl_s1`, seed 1) is trained.
   - The rank-2 graft of (`ctrl_s1` − `ctrl`) onto `ctrl`, run iteratively (`g_noise_r2`), must also recover less than half of the real rank-2 graft on the tuning set.
   - This rules out "any training-run difference has helpful leading directions", which random matrices alone do not.
2. **Denominator stability:** "holds" also requires the 95% paired-bootstrap CI of the test gap (`iter_iter` − `ctrl_iter`) to be above 0.
3. **Outcome classes, fixed now:**
   - **holds**
   - **fails (control)**
   - **inconclusive:** point estimate ≥ 0.8 but CI lower bound ≤ 0.5
   - **fails:** point estimate < 0.8
   - **not testable:** the gate fails, or the test-gap CI is not above 0
4. **Replication rule:**
   - **Only "holds on both datasets" counts as replication.** That is a conjunctive claim, so no multiplicity correction is needed.
   - A single dataset holding is reported as exploratory.
5. **Integrity checks:**
   - the HotpotQA exclusion of exp01–exp07 ids is asserted in `analyze.py`
   - the notebook asserts a T4 and prints torch/transformers versions; each evaluation line records the GPU and model hashes
   - the training parquet files are pinned to immutable Hugging Face commits, and their sha256 is printed and recorded at run time (they were not downloaded locally, so a hash could not be fixed in advance)
6. **Descriptive additions:**
   - the fraction of iterative queries reaching 512 tokens (truncation)
   - recovery at fixed recall@5 (the same k on every dataset)
7. **Scope:** HotpotQA is reported as *adaptation of A1*, since A1 was fine-tuned on HotpotQA train. 2WikiMultiHopQA, which A1 never saw, is the cleaner replication. The claim is about the rank-2 component of one fitted delta per dataset (seed 0), checked against random and seed-noise nulls.
8. **Power:** with 1,000 test questions and an expected gap of about 0.3 (MuSiQue: 0.34), a rank-2 recovery of 0.87 would have a CI of about ±0.09, comfortably above 0.5. A gap near 0.05 would make the ratio unstable, which the gap-CI rule guards against.

## Deviation 1 (before any training): A1 location

The free GPU quota of the first Google account was exhausted, so the run uses a second account. The notebook now looks for A1 in either `exp02_results/full/model` (first account) or `exp12_inputs/a1` (a copy placed earlier on the second account's Drive). The **sha256 check of A1 is unchanged** (`605c98…950d`). The experiment code is still the registration commit 353fcd8; only `make_notebook.py` / `exp16_colab.ipynb` changed.
