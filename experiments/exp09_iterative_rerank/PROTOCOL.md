# exp09 protocol: iterative (query-updated) reranking on MuSiQue-Ans (pre-registered 2026-11-13)

## Question

exp08 showed that on MuSiQue the **one-shot ranker** is the bottleneck. A1's top 10 holds all supporting paragraphs for only 0.67 of test questions, and its top 5 only 0.35; at the 0.90-tuned A1 threshold, 4-hop recall was 0.67. Later hops do not look like the question.

**Does re-scoring the remaining paragraphs with "question + paragraphs selected so far" put the full chain higher than one-shot scoring does, when both rankers are trained on the same MuSiQue questions?**

## Data (fixed in `split_ids.json` before any training; `split_ids_make.py`)

- **train:** 2,000 MuSiQue-Ans **train** questions (seed 9), disjoint from exp08's pool.
- **dev300:** the first 300 ids of exp08's dev; used only to pick k for the token comparison.
- **test:** exp08's 400 test questions (all of them from MuSiQue-Ans dev).
- Files are read from `../exp08_musique/data/`. The paragraph format is `"<title>. <text>"`, as in exp08.

## Rankers (four arms)

| Arm | Model | Inference |
|---|---|---|
| `A1` | A1 (HotpotQA), frozen | one-shot: score (q, p) once, sort. exp08's cache |
| `A1-iter` | A1, frozen | iterative (below), zero-shot |
| `M1` | A1 fine-tuned on MuSiQue **step-0** examples only | one-shot |
| `M1-iter` | A1 fine-tuned on MuSiQue examples of **all steps** | iterative |

**Training (M1 and M1-iter)**
- **Base:** initialised from A1. Pointwise BCE, AdamW lr 2e-5, 1 epoch, batch 16, max length 512, seed 0, CPU.
- **Example at step t** (t = 0 … h−1, with h = the number of hops):
  - query = question + the first t gold paragraphs in decomposition order (`paragraph_support_idx`), most recent first, joined with " [SEP] "
  - positives = **all** gold paragraphs not yet in the query (order-agnostic)
  - negatives = 6 non-gold paragraphs drawn uniformly (seeded per question and step)
  - truncation: the query side only (`only_first`), so the candidate paragraph is never cut
- **Same data for both:** `M1` gets exactly the step-0 examples of `M1-iter` (same questions, same negative draw). Its query is the question alone. `M1-iter` therefore sees more pairs (≈ h× as many); that is the method, and its training cost is reported.

**Iterative inference (greedy)**
- Step 0 scores every paragraph with the question alone and takes the top one.
- Step s scores the remaining paragraphs with the query (question + the selected paragraphs, most recent first) and takes the top one.
- After **4 greedy steps** (MuSiQue has at most 4 hops), the remaining paragraphs are ordered by their step-4 scores.
- The result is a full ranking of all paragraphs. Cost: ≈ 20 + 19 + 18 + 17 = 74 cross-encoder passes per question (vs 20 for one-shot), reported.

## Metric

- **Full recall@k:** all supporting paragraphs (2–4) are in the top k of the ranking.
- **Tokens:** the exp08 prompt with the top k paragraphs (tiktoken `o200k_base`).

## Pre-registered comparisons (test, 400)

1. **Primary (does iteration help, at equal training data?):** full recall@5, `M1-iter` vs `M1`, McNemar exact, two-sided α = 0.05. **Success** = M1-iter higher and p < 0.05.
2. **Secondary:**
   - a. full recall@5, `M1-iter` vs `A1` (the exp08 ranker), McNemar.
   - b. **Tokens at a recall target:** for each arm, k = the smallest k with dev300 full recall ≥ 0.90 (and ≥ 0.95), applied to test. Report test recall and mean tokens. Paired bootstrap CI of the token difference `M1-iter` − `M1` and `M1-iter` − `A1`. The recall CI must include 0 or be above it, as in exp07/08.
3. **Descriptive:**
   - recall@k curves, k = 2 … 10, all arms
   - results by hop count
   - `A1-iter` (does iteration work zero-shot?)
   - cross-encoder passes per question
   - training time

## Budget rule (fixed now)

A smoke test measures training throughput first. If the projected total CPU time (training both models plus scoring dev300 and test) exceeds 10 h, train on the **first 1,000** of the 2,000 train ids instead. This is recorded either way.

## Honest priors

- **Gain likely.** Iterative retrieval is a known remedy for multi-hop recall (e.g. multi-hop dense retrieval, IRCoT). Here the step-0 ranker is weak, so a single bad first pick limits the greedy chain.
- **M1 alone may close much of the gap.** In-domain fine-tuning is the obvious confound, which is why M1 is the primary comparison, not A1.
- **A1-iter zero-shot may get worse.** A1 never saw long queries.
- **4-hop questions:** the most to gain, but the fewest examples (64 in test).
