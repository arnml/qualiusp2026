# exp06: how small is the multi-hop skill? (A1 − A0 grafting)

Pre-registered in `PROTOCOL.md` (commit df5c61f). Code: `graft.py`, `common.py`. Raw results: `results/grafts_dev300.jsonl`, `results/grafts_dev1000.jsonl`.

**Recovery** = the share of A1's tok@90 gain that a graft recovers: `(tok90(A0) − tok90(graft)) / (tok90(A0) − tok90(A1))`. The bg3 recovery is defined the same way.

## Answer in one line

**A1's skill is low-rank and spread out, not sparse.**
- Rank 4 per matrix (328k parameters) recovers **80%** of A1's gain.
- The 108k largest entries within Astra's scope recover **3.5%**.
- So R1's null result is explained by the *shape* of Astra's budget (a cap on the number of changed entries), not only by Astra's search.

## Pre-registered readings (dev1000)

| # | Reading | Result | Verdict |
|---|---|---|---|
| 1 | `astra_scope_top_108k` recovers ≥ 50% of the tok@90 gain | **3.5%** (bg3 6.0%) | **Under 20%: the cap explains the R1 null** |
| 2 | Magnitude beats random at the same size by more than 2 SE | 1% graft: bg3 0.639 vs 0.599 (+0.040, ≈ 1.8 unpaired SE); tok@90 recovery 7.6% vs 0.3% (no SE: per-question scores were not saved) | **Not shown at 2 SE** on bg3. The direction favours magnitude, but both recoveries are tiny. |
| 3 | Smallest rank with ≥ 80% recovery | **Rank 4: 80.4%** (328,425 parameters; bg3 0.885 vs A1 0.959). Rank 8: 91.6%. | Rank 4, just over the bar |

## Sparse vs low-rank at similar budgets

| Budget | Sparse (top-\|Δ\|) | Low-rank |
|---|---|---|
| ~0.1M | 108k in Astra's scope: 4.6% (dev300), 3.5% (dev1000) | rank 1 (106k): 16.4% |
| ~0.2M | 227k (1%): 9.9% (dev300), 7.6% (dev1000) | rank 2 (180k): 65.2% (dev300), 64.7% (dev1000) |
| ~0.3M | (not run) | rank 4 (328k): 79.5% (dev300), 80.4% (dev1000) |
| ~2.3M | 2.27M (10%): 76.7% | rank 32 (2.4M): 98.3% |

At equal parameter budgets, low-rank needs about **10× fewer** parameters than sparse for the same recovery.

**Caveat (practical):** a low-rank graft changes *every* entry of each matrix once it is merged. It saves storage and training parameters, **not inference compute**.

## Where the gain lives (dev300; descriptive, no decision rule)

| Graft only… | Parameters | Recovery | | A1 with one part reverted | Recovery |
|---|---|---|---|---|---|
| biases / LayerNorm | 31k | 1% | | without embeddings | 99.5% |
| embeddings | 11.9M | 0% | | without head | 99.7% |
| head (pooler + classifier) | 148k | 0% | | without L0 | 98.2% |
| all attention | 3.6M | **55%** | | without L1 | 98.1% |
| all MLP | 7.1M | **71%** | | without L2 | 96.5% |
| L0 | 1.8M | 3% | | without L3 | **87.4%** |
| L1 | 1.8M | 5% | | without L4 | **83.0%** |
| L2 | 1.8M | 11% | | without L5 | 97.6% |
| L3 | 1.8M | 22% | | | |
| L4 | 1.8M | **36%** | | | |
| L5 | 1.8M | 3% | | | |

**Reading:**
- The skill sits mostly in the **middle layers, L3–L4**, in both attention and MLP.
- Embeddings (half of all changed parameters), the head and the last layer contribute almost nothing.
- **The parts are redundant and interact:** attention-only and MLP-only each recover more than half, and no single layer is necessary. A "graft only" result is not the same as a "leave out" result.

## Limits

- **Selection:** grafts are chosen with A1 (an oracle on the delta). They show that a small repair *exists* in a given shape, not that one can be found from A0 alone.
- **Model:** a single model pair (MiniLM-L6) and one dataset.
- **Recovery measures:** tok@90 is a threshold metric; no SE was computed for it.
- **Sweep size:** dev300 numbers for the components are a sweep, not a confirmation.

## What follows (for NEXT_DIRECTIONS)

- **Astra repair v2:** give the editor a **rank budget** (e.g. rank ≤ 4 updates to L3–L4 matrices) instead of an entry cap. This would also need a decision on whether gradient fitting is allowed (open question to the PI).
- **Pre-registered step 2 is not triggered:** reading 1 failed, so Astra-as-graft-selector under the 108k cap is not run.
