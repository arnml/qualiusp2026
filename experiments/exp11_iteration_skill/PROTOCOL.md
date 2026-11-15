# exp11 protocol: is the iteration skill low-rank and localized? (pre-registered 2026-11-14)

## Question

exp06 found that A1's HotpotQA skill (A1 − A0) is **low-rank** (rank 4 per matrix recovers 80% of the gain) and sits mostly in **layers L3–L4**. exp09 trained two new skills on MuSiQue, both starting from A1:

| Skill | Delta | Base → target | Inference |
|---|---|---|---|
| **Iteration** (primary) | D_it = M1-iter − M1 | M1 → M1-iter | iterative (exp09: 4 greedy steps, tail ordered by the step-3 scores) |
| **Domain** (secondary) | D_dom = M1 − A1 | A1 → M1 | one-shot |

1. **Is the iteration skill low-rank, like the HotpotQA skill?**
2. **Does it sit in the same layers (L3–L4)?**
3. Same two questions for the domain skill, as a comparison.

**Why it matters:** if chain-following in a 22M cross-encoder is a small, low-rank change in a few middle layers, then:
- the skill has a specific location, which is a mechanistic claim
- it can be shipped as a tiny adapter on a shared model (practical)

**Known so far (novelty boundary):** that fine-tuning deltas are often low-rank is established (intrinsic dimension, Aghajanyan et al. 2021; LoRA, Hu et al. 2022; task arithmetic, Ilharco et al. 2023). The new part tested here is *which* skill (iteration over a chain vs domain adaptation), *where* it sits, and whether it matches exp06.

**Note on D_it:** M1 and M1-iter were fine-tuned separately from A1, so D_it also contains differences between the two training runs (M1-iter saw about 2.2× more pairs). By construction M1 + D_it = M1-iter exactly, so recovery is well defined. It measures "the change that turns M1 into M1-iter", not a pure iteration direction.

## Method (no data used to choose a graft)

`graft.py`. Each configuration is base + (part of the delta), scored with that family's inference. The models are frozen exp09/exp04 checkpoints.

| Family | Configurations (for both `it_` and `dom_`) |
|---|---|
| **Endpoints** | `base` (for `it`: M1 run **iteratively**), `full` (= target; a sanity check that must reproduce exp09) |
| **Low-rank** (exp06: truncated SVD of each 2-D delta, 1-D deltas in full) | rank 1, 2, 4, 8, 16 |
| **Random control** | random rank-4 matrices, each scaled to the Frobenius norm of the rank-4 SVD part (seed 0) |
| **Localization, graft only** | each layer L0–L5 (attention + MLP of that layer only) |

That makes 14 configurations per family. The `it` family is the expensive one: 74 passes per question, about 25 min per configuration on dev300.

## Evaluation

- **Sweep:** dev300 (exp09/exp10's dev split), all configurations.
- **Confirmation:** test (exp09's 400). For each family, run `base`, `full` and the dev300-chosen r* once.
- **Metric:** full recall@5 (exp09's primary metric). Recall@k for k ∈ {2, 3, 4, 5, 6, 8, 10, 15} and per-question hits are stored.
- **Recovery:** `(R(graft) − R(base)) / (R(full) − R(base))`, using recall@5.

## Pre-registered readings

**Gate:** if the dev300 gain R(full) − R(base) for `it` is below **0.05**, recovery is too noisy to read. In that case reading 1 is **not testable** and is reported descriptively.

1. **Primary, "the iteration skill is low-rank":**
   - r* = the smallest rank with **≥ 80%** recovery on dev300 (the exp06 bar).
   - **Holds if both:**
     - r* ≤ **8**
     - on test, the paired-bootstrap 95% CI of recovery for r* has a lower bound **> 0.5**
   - **Fails if** no rank ≤ 16 reaches 80% on dev300.
2. **Control:** `it_rand_r4` must recover **less than half** of what `it_r4` recovers on dev300. Otherwise "low-rank" cannot be told apart from "any perturbation of that size".
3. **Localization, "same layers as HotpotQA":** for each family, holds if **L3 and L4** are the two layers with the highest graft-only recovery on dev300. Otherwise the per-layer profile is reported and compared with exp06.
4. **Domain skill:** r* and the localization, as in readings 1 and 3. This is descriptive: a contrast to the iteration skill.

**Tests on test:** for r*, McNemar exact on hit@5 against `base` and against `full`.

## Budget rule

- After the first `it` configuration, project the total time. If it is over **12 h**, drop configurations in this order: `it_r16`, `it_L0`, `it_L5`. Record any drop in `results/budget.json` before continuing.
- `dom` configurations are cheap (one pass of about 20 per question) and are always run.

## Honest priors

- **Low rank:** likely. exp06 found rank 4 for a bigger change (A0 → A1, 10k HotpotQA steps). D_it comes from about 1,100 steps on the same base.
- **Layers:** unclear. Reading a long query ("question [SEP] picked paragraph") could need earlier layers to bind the bridge entity, or the head.
- **Possible failure mode:** M1 run iteratively might already do well (unlike A1-iter, 0.12). That would leave a small gain; the gate covers it.

## Code check before registration

`graft.py --smoke 3` was run on 3 dev300 questions only, to check that every config family builds and ranks. The output was deleted and is not used. Speed: about 3.5 s per question for iterative configs, about 18 min per `it` configuration, about 5 h for everything.
