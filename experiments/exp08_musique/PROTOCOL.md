# exp08 protocol: the exp07 adaptive selector on MuSiQue (pre-registered 2026-11-13)

## Question

**Does exp07's chain-aware, adaptive selection carry over to MuSiQue?** exp07 was positive on HotpotQA: −57 to −71 tokens at recall 0.90/0.95, with F1 not significantly changed. MuSiQue is the stronger test:
- 2–4 hops instead of 2
- 20 candidate paragraphs instead of 10
- built so that single-hop shortcuts fail

## Data

- **Source:** MuSiQue-Ans v1.0 (`bdsaglam/musique` mirror; files in `data/`, git-ignored).
  - The official test split has no labels, so the labelled dev split (2,417 questions) provides dev and test.
  - Paragraph-level labels: `is_supporting`.
- **dev (1,000) and test (400):** a seeded (seed 0) random, disjoint draw from MuSiQue-Ans dev. Ids are saved to `split_ids.json` before any scoring.
- **pool (4,000):** a seeded random sample of MuSiQue-Ans **train**, used to train the stopping scorers (80% train / 20% early stopping).
- **No MuSiQue data trained A1.** A1 is the HotpotQA-fine-tuned MiniLM-L6 cross-encoder from exp04, used frozen (a transfer setting).

## Scoring (cache)

`score.py`: A1 scores every (question, paragraph) pair once, keeping the score and the 384-d pooler vector, as in exp05. The cache goes to `cache/` (git-ignored).

## Target and metric

- **"Full recall":** *all* supporting paragraphs (2–4) are in the selected set.
- **Tokens:** prompt tokens of the exp02/exp07 retrieval prompt (tiktoken `o200k_base`). The selected paragraphs are shown in A1 score order.
- **Paragraph format and prompt:** each paragraph is formatted as in exp02, `"<title>. <text>"`. The prompt is the exp02/BRACIS prompt with one change: "Answer the HotpotQA question" becomes "Answer the question".

## Policies

| Policy | Paragraphs |
|---|---|
| **A1 threshold** (exp02 rule) | score ≥ t (at least 1) |
| A1 top-k | fixed k (descriptive) |
| **adaptive** | A1's top k, for the smallest k ∈ {2, …, 10} with p_k ≥ τ; else k = 10 |

**p_k** = the predicted probability that A1's top k contains all supporting paragraphs. Three stopping scorers (the same split of roles as exp07), trained on the pool with one example per (question, k):

| Scorer | Model | Features |
|---|---|---|
| `a1only` | logistic regression | k, A1's top-10 sorted scores, the gap s_k − s_{k+1} |
| `text` | logistic regression | `a1only` + chain/link features (below) |
| `full` | 64-unit MLP | `text` + pooled vectors: mean and max of h over the top k, and h of paragraph k+1 |

**Chain/link features**, from titles and text only (no extra model passes):
- number of top-k paragraphs whose title appears in another top-k paragraph
- whether a top-k paragraph mentions the title of a paragraph *outside* the top k (a "dangling hop"), and of paragraph k+1 in particular
- number of top-k titles that appear in the question
- word overlap (Jaccard) between the question and the top-k union, and between the question and paragraph k+1

**Tuning:** τ is chosen on dev (smallest mean tokens with dev recall ≥ target), and so is t for the A1 threshold. Both are applied **once** to test.

## Pre-registered comparisons (test, 400)

Targets: full recall **0.90** and **0.95**.

1. **Primary (practical):** `adaptive` (`text` or `full`) vs `A1 threshold`.
   - The paired bootstrap 95% CI of the token difference must lie entirely below 0.
   - The recall difference CI must include 0 or be above it.
   - **Success** = both, for at least one scorer at at least one target. This is the exp07 rule.
2. **Secondary (chain value, pre-registered here; exploratory in exp07):** `adaptive_full` (and `text`) vs `adaptive_a1only`, with the same criteria. **Success** = token CI below 0 with recall not worse, at either target.
3. **Descriptive:**
   - each method's best threshold on test itself (equal recall, the same procedure for all)
   - mean paragraphs sent
   - results by hop count (2 / 3 / 4)
   - A1 top-k recall curve
4. **Follow-up, only if 1 succeeds:** gpt-5.4-mini answer F1 on the 400 test questions.
   - Conditions: the primary winner vs A1 threshold, at both targets (≈ 1,600 calls).
   - Prompt and F1 as in exp02/exp07. F1 is the max over `answer` and `answer_aliases`.

## Honest priors (stated before any scoring)

- **A1 transfer is uncertain.** A1 never saw MuSiQue, and MuSiQue's distractors are harder (built from the same entities). Recall at small k may be low, and the A1 threshold may need many paragraphs to reach 0.95.
- **More room than on HotpotQA.** 20 paragraphs (~2,000 tokens) with 2–4 needed leaves more room for adaptive savings.
- **Chain features may matter more than in exp07.** Later hops are less similar to the question, so link features may help, especially on 3–4 hop questions.
- **The gold set is not the only answer path.** A question can sometimes be answered without every supporting paragraph, so recall is a conservative proxy for F1.
