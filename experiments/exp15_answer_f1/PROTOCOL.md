# exp15 protocol: does iteration's recall gain on long chains reach the answers? (pre-registered 2026-11-15)

## Why

exp13 confirmed a recall@5 gain from conditioning-trained iteration on 4-hop chains. Exploratorily, the gain was +0.078 on 3-hop and +0.139 on 4-hop of fresh dev questions, and about 0 on 2-hop. Recall is a proxy. **The practical question is answer quality** at the same context size (top 5).

## Data and contexts (no new ranking)

- **Questions:** exp13's `dev_fresh` (1,017 unused MuSiQue-Ans dev questions; 529 / 322 / 166 two/three/four-hop).
- **Top-5 orders:** exp13's frozen models, copied from the Colab Drive into `results/top5_dev_fresh.json`. **Checked before registration:** hit@5 recomputed from the copy equals exp13's recorded hits on every question, for both models. Model sha prefixes: `ed87244f…` (M1-ctrl), `2c91c40f…` (M1-iter).
- **Conditions (context = paragraphs in this order):**

| Condition | Context |
|---|---|
| `ctrl_top5` | M1-ctrl one-shot top 5 |
| `iter_top5` | M1-iter iterative top 5 (greedy pick order, then tail) |
| `gold` | the supporting paragraphs only (ceiling; descriptive) |
| `all` | all ~20 paragraphs (exp08's "send everything" reference; descriptive) |

- **Reader:** gpt-5.4-mini with exp02/exp08's prompt and settings (default sampling).
- **Metric:** answer F1 = max over `answer` and `answer_aliases`, exp02's normalization.
- **Calls:** 4 × 1,017 = 4,068, 4 concurrent workers. Estimated cost about $0.9 (`all` about $0.5).

## Pre-registered tests (paired bootstrap, 5,000 resamples, seed 0)

1. **Primary:** on ≥ 3-hop questions (n = 488), mean F1(`iter_top5`) − F1(`ctrl_top5`).
   - **Success** = the 95% CI lower bound is > 0.
   - **Power (rough):** exp02 found that a missing gold paragraph costs about 0.30 F1. A +0.10 recall@5 gain therefore suggests about +0.03 F1. With a per-question SD of about 0.35, the SE is about 0.016, so power is roughly 50%. **The CI is the main evidence.**
2. **Secondary (labelled secondary):**
   - **a.** All questions: the same difference with its CI.
   - **b.** 2-hop non-inferiority: the one-sided 95% lower bound of the difference must be > −0.02.
3. **Descriptive:**
   - F1 and input tokens for all four conditions, by hops
   - `gold` (ceiling) and `all` (reference)

**Claim ledger:** primary = H13 ("the long-chain recall gain reaches the answers").

## Re-use disclosure

`dev_fresh` was the cascade test set in exp13, and its per-hop recall results motivated this test. F1 has never been computed on it, and no choice here depends on answers.

## Honest priors

- **Primary:** uncertain. The reader may answer from partial evidence, or fail even when all the evidence is present.
- **2-hop non-inferiority:** likely, since recall@5 is about equal there.
- **`all` may beat top-5 on F1** at about 4× the tokens. That is the cost trade-off this project studies.

## Changes after the adversarial pre-run review (Codex, `../reviews/codex_exp15_review.md`; before any call)

1. **The primary is order-controlled.** It compares `ctrl_idx` and `iter_idx`: each model's top-5 *set*, presented in dataset-index order, so only *which* passages are included differs.
   - **Success** = the 95% CI lower bound of mean F1 difference (iter − ctrl) on ≥ 3-hop questions is > 0.
   - The ranker-order versions (`ctrl_top5`, `iter_top5`) are secondary: the policy effect.
2. **Reader variance:**
   - `ctrl_idx` and `iter_idx` get a **second reader draw** on every ≥ 3-hop question, and the primary uses the mean of the two draws
   - draw agreement is reported
   - all conditions of a question are **submitted together** (interleaved), so condition is not confounded with time
   - sampling stays at the API default, and the returned model id is recorded per call
3. **Integrity checks:**
   - the top-5 copy is asserted: length, index range, first and last IDs
   - the report rejects duplicate IDs, wrong condition labels and mismatched paragraph lists, and requires every answer
   - blank answers are counted
4. **Reported per condition:** EM next to F1, and input-token mean / p10 / p90 (the token budget is not matched; it is reported).
5. **Calls:** 6 × 1,017 + 2 × 488 = 7,078, about $1.3.
6. **Framing:** the ≥ 3-hop subgroup was chosen from exp13's recall on this same set. The claim is therefore "on this set, the recall gain reaches the answers". All secondaries are exploratory.
