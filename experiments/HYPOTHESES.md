# Claim ledger (started 2026-11-15)

**Rule:** only pre-registered primary tests are **confirmatory**. Everything else is **exploratory** and must be re-tested in a new pre-registered experiment before it is claimed.

**Status:**
- **C** = confirmed by a pre-registered primary test
- **C?** = confirmed but confounded or not replicated
- **N** = null or refuted
- **E** = exploratory only
- **K** = known in the literature (see `LITERATURE_POSITIONING.md`)

| ID | Claim | Exp | Status | Replicated? | Open threat |
|---|---|---|---|---|---|
| H1 | A fine-tuned 22.7M selector cuts LLM tokens 68–77% at equal F1 vs top-10 (HotpotQA) | exp02 | C, K in spirit | no | one dataset; no Adaptive-k baseline |
| H2 | LLM weight editing compresses A1 but ties Wanda; it does not repair A0 | exp04 | C (tie), N (repair) | Opus replicates | one editor per track |
| H3 | The A1 − A0 delta is low-rank (rank 4 recovers 80%), not sparse; it sits in L3–L4 | exp06 | C, mostly K | no | magnitude-sparse is a weak baseline; one model |
| H4 | A learned adaptive stop saves tokens vs a threshold (HotpotQA) | exp07 | C, K | no | Adaptive-k: weak in BRACIS (cosine, F1 0.627) and on our cross-encoder scores (exp10 addendum, E) |
| H5 | On MuSiQue the one-shot ranker is the bottleneck | exp08 | E (primary not testable) | – | – |
| H6 | An iterative ranker beats one-shot at recall@5 (0.750 vs 0.682) | exp09 | **C?** | no | **2.2× more training pairs (exp12)**; one seed |
| H7 | Adaptive stopping beats fixed k on M1-iter at recall 0.95 | exp10 | C at 0.95, N at 0.90 | no | dev300 τ is noisy |
| H8 | The iterative score is a better stop signal than one-shot | exp10 | **N** | – | – |
| H9 | The iteration change (M1 → M1-iter) is low-rank (r* ≤ 8) | exp11 | **C?** (rank 2: 84.7% [74, 96] on test) | no | confounded like H6 (exp12 secondary); one seed |
| H10 | The iteration change sits in L3–L4 like H3 | exp11 | C (dev300 rule), flatter profile | no | keep-only grafts overlap (sum ≈ 180%) |

**Multiplicity:** about 12 confirmatory primaries so far, and every one was a declared single primary per experiment. No correction has been applied. For the thesis, the headline claims (at most 3) will be re-confirmed on a locked set.
