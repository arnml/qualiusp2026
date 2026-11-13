# exp07 protocol: a chain-aware, adaptive evidence selector (pre-registered 2026-11-13)

## Questions (from the PI)

1. Can a small model judge whether two paragraphs **complete each other**, rather than score each alone?
2. Can retrieval be **adaptive**: 2 paragraphs when the evidence pair is clear, 3–4 when it is not?

## Design (cheap by construction)

- **No extra cross-encoder passes.** A1 scores the 10 paragraphs once, as today.
- **Pair scorer** over the 6 pairs of A1's top 4, using A1's cached scores and 384-d pooler vectors plus cheap text features:
  - does each paragraph's title appear in the other paragraph?
  - does each title appear in the question?
  - word overlaps, and shared capitalised words
- **Two scorers:**
  - `text`: logistic regression on A1 scores + text features
  - `full`: a 64-unit MLP that also uses the vectors
- **Training data:** the exp05 pool (4,000 unused validation questions; 80% train / 20% early stopping). A1 never trained on these questions.
- **Code:** `chain.py`. Inputs come from `../exp05_failure_anatomy/cache`.

### Policies

| Policy | Paragraphs |
|---|---|
| A1 top-k | 2 / 3 / 4 |
| A1 threshold (exp02 rule) | score ≥ t |
| pair_best | the best pair (2) |
| **adaptive** | best pair if p ≥ τ2; best pair + next A1 paragraph if p ≥ τ3; otherwise A1 top 4 |

## Pre-registered comparisons

Thresholds are tuned on **dev** and applied **once** to **test** (400).

1. **Primary (practical):** at recall targets 0.90 and 0.95 (both gold kept), the test tokens of `adaptive` vs `A1 threshold`.
   - The paired bootstrap 95% CI of the token difference must be entirely below 0.
   - The recall difference CI must include 0 or be above it.
   - **Success** = both, for at least one scorer.
2. **Secondary (multi-hop):** with exactly 2 paragraphs, does `pair_best` keep both gold more often than A1 top-2? Exact McNemar on test, p < 0.05.
3. **Descriptive:** the oracle-on-test frontier (each method's best threshold on test itself, the same procedure for both), and the mean number of paragraphs sent.
4. **Follow-up, only if 1 or 2 succeeds:** gpt-5.4-mini answer F1 on the 400 test questions for the winner vs A1 threshold (exp02 `generate.py` pattern, ~800 calls).

## Honest priors

- **Headroom on HotpotQA distractor is small.** A1 keeps both gold in its top 3 for ~94–96% of questions, and the oracle needs 323 tokens vs A1's ~390 at 90%. The best realistic gain is tens of tokens per question.
- **exp05's category A (bridge-linked misses)** tells how many failures a title-link feature could fix.
- **The stronger test of chain awareness** is MuSiQue / 2Wiki (planned next, not part of this protocol).
