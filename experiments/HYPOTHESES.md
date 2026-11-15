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
| H6 | An iterative ranker beats one-shot at recall@5 (0.750 vs 0.682) | exp09, exp12 | **N at matched data** (exp12: +0.040 [−0.005, +0.085], p = 0.11; +0.03 in all 3 seeds) | 3 seeds | 4-hop +0.19 (p = 0.036) is E, needs re-test |
| H7 | Adaptive stopping beats fixed k on M1-iter at recall 0.95 | exp10 | C at 0.95, N at 0.90 | no | dev300 τ is noisy |
| H8 | The iterative score is a better stop signal than one-shot | exp10 | **N** | – | – |
| H9 | The iteration change is low-rank (r* ≤ 8) | exp11, exp12, exp16 | **C** (D_it rank 2: 84.7% [74, 96]; pure conditioning D rank 2: 86.8% [78, 96]) | 2 deltas | one seed per delta; spectral and noise baselines owed |
| H10 | The iteration change sits in L3–L4 like H3 | exp11 | C (dev300 rule), flatter profile | no | keep-only grafts overlap (sum ≈ 180%) |
| H6-4hop | On ≥ 4-hop chains, conditioning-trained iteration beats matched one-shot | exp13 | **C** (+0.055 [+0.007, +0.102], p = 0.035, n = 566; strata +0.139 / +0.020) | 3 seeds positive | heterogeneous; 3-hop gain on fresh dev is E |
| H11 | A 4-hop router on one-shot features makes a useful cascade | exp13 | **N** (vs one-shot +0.009 n.s.; vs iterative −0.043) | – | the router predicts hop count, not iteration benefit |
| H14 | The pure conditioning change is rank 2 on other datasets (2WikiMultiHopQA, HotpotQA) | exp16 | **C** (2Wiki 97.1% [94.1, 100]; HotpotQA 92.1% [81.6, 102.8]; random 3% / 22%, noise −3%) | 3 datasets with exp12 | one model family; one control draw on tuning data |
| H12 | A router trained on iteration benefit gives a useful cascade (beats one-shot at ≤ 50 passes) | exp14 | **C** (+0.027 [+0.016, +0.039], p = 5e-6, 40.8 of 74 passes; local re-run +0.023, p = 1e-4) | router refit in 2 environments | models re-trained (behaviour identical on exp12 test); train-distribution sample, mostly 2-hop; non-inferiority to always-iterative is fragile (−0.0065 vs −0.0105) |
| H13 | The long-chain recall gain reaches the answers (F1, ≥ 3-hop, order-controlled) | exp15 | **C** (+0.076 [+0.041, +0.112]) | 2 reader draws | subgroup chosen from exp13 recall on the same set |

**Multiplicity:** about 12 confirmatory primaries so far, and every one was a declared single primary per experiment. No correction has been applied. For the thesis, the headline claims (at most 3) will be re-confirmed on a locked set.
