# exp11: is the iteration skill low-rank and localized? (delta grafting)

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit bf71534).
- **Grafts:** parts of a fine-tuning delta are grafted onto its base model, with no data used to choose them.
- **Sweep:** dev300, 28 configurations. **Confirmation:** test (400), endpoints plus r*.
- **Metric:** full recall@5. Recovery = (R(graft) − R(base)) / (R(full) − R(base)).
- **Code:** `graft.py` (sweep and test), `analyze.py` → [`results/results.json`](results/results.json).

## Bottom line

1. **Primary holds: the iteration change is rank 2.** M1 plus a rank-2 SVD of D_it = M1-iter − M1 (per matrix, about 180k parameters) recovers **84.7% [73.6, 96.2]** of the iterative gain on test. That is recall@5 0.7075, between base 0.4725 and full 0.75.
   - **The control passes:** a norm-matched random rank-4 change recovers 9% on dev300, against 97% for the real rank-4 change.
   - **Not lossless:** rank 2 stays below full M1-iter (14 vs 31 discordant questions, p = 0.016).
2. **The skill sits in the middle layers.** L3 (43%) and L4 (40%) are the top single layers on dev300, so the pre-registered "same layers as HotpotQA" holds. The profile is flatter than exp06's: L0–L2 each give about 30%, and only L5 gives almost nothing (8%).
3. **The domain change (A1 → M1) is less compressible.** Its best rank on dev300 is also 2, but exactly at the bar (76 of 95 questions). On test, rank 2 recovers **72.5% [63.2, 81.4]**, and its top layers are L3 and L2. This comparison is descriptive.
4. **The main caveat comes from exp12.** D_it mixes "conditioning on picked paragraphs" with "2.2× more training". exp12 shows that much of exp09's iterative gain was more training. Its secondary test asks whether the *pure* conditioning change (M1-iter − M1-ctrl) is also rank 2.

## Numbers

**dev300 recovery** (iteration: base 0.45, full 0.74, gain 0.29; domain: base 0.393, full 0.71, gain 0.317):

| Graft | Iteration (D_it) | Domain (D_dom) |
|---|---|---|
| rank 1 | 74% | 64% |
| **rank 2** | **95%** | **80%** |
| rank 4 | 97% | 84% |
| rank 8 | 97% | 103% |
| rank 16 | 91% | 101% |
| random rank 4 | 9% | 6% |
| L0 / L1 / L2 | 33 / 31 / 30% | 16 / 21 / 37% |
| **L3 / L4** | **43 / 40%** | 42 / 28% |
| L5 | 8% | 9% |

**Test confirmation** (recall@5):

| Family | Base | Rank 2 | Full | Recovery [95% CI] | Rank 2 vs base | Rank 2 vs full |
|---|---|---|---|---|---|---|
| Iteration (iterative inference) | 0.4725 | 0.7075 | 0.75 | **0.847 [0.736, 0.962]** | 109 / 15, p = 9e−19 | 14 / 31, p = 0.016 |
| Domain (one-shot) | 0.355 | 0.5925 | 0.6825 | 0.725 [0.632, 0.814] | 104 / 9, p = 1e−21 | 6 / 42, p = 1e−7 |

## Reading, with caveats from the literature re-analysis

- **Rank-to-rank differences on dev300 are within noise.** One recovery point is about 0.9 questions, so ranks 2–8 form a plateau, not an ordering.
- **"Rank 2" here is functional, not spectral.** It is the per-matrix truncation that keeps the effect. Still owed:
  - singular-value energy curves and stable rank
  - a noise baseline: more random draws, plus a seed-noise delta, which exp12 provides
  - principal angles between D_it, D_dom and A1 − A0 (Altaf 2026; PHLoRA)
- **Prior work bounds the novelty:**
  - a behavior living in a rank-1 change is shown for LLMs (rank-1 LoRA in Soligo, Turner et al. 2025; TinyLoRA under RL)
  - what is new here is a *retrieval* behavior in a 22M reranker, measured against a domain change in the same model
- **Single-layer recoveries add up to about 180%,** so the layers overlap. "Keep only Lx" does not say which layers are *necessary* (exp06 had the leave-one-out view).

## Deviations

- **The sweep was interrupted** by a machine restart after 19 of 28 configurations and resumed (per-configuration resume). No configuration was affected.
- **The analysis script (`analyze.py`) was written after the dev300 sweep** but before any test result. It implements the PROTOCOL rules as stated.

## Files

- `PROTOCOL.md`, `split_ids.json`, `graft.py`, `analyze.py`, `data.py`
- `results/grafts_dev300.jsonl`, `results/grafts_test.jsonl`: per-question hits and recall@k
- `results/results.json`
