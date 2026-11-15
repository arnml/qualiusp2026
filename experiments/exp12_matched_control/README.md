# exp12: pair-matched control for the iterative ranker

Pre-registered in [PROTOCOL.md](PROTOCOL.md): commit 975371a, revised after an adversarial pre-run review by Codex.
- **Control:** M1-ctrl is trained on M1-iter's **exact** 18,101 training pairs in the same order, with the query prefix of picked paragraphs removed.
- **Compute:** Colab T4, seeds 0–2. Code pinned to 975371a. The analysis code (`analyze.py`) was written before training.
- **Results:** [`results/results.json`](results/results.json) and per-question hits in [`results/hits_compact.json`](results/hits_compact.json). The raw orders stay on the Colab account's Drive.

## Bottom line

1. **Primary fails: once training data is matched, iteration's overall gain is small and not significant.** M1-iter (iterative inference) vs M1-ctrl (one-shot), test recall@5:
   - **0.7475 vs 0.7075, +0.040 [−0.005, +0.085]**, 52 vs 36 discordant questions, p = 0.109.
   - The minimum detectable effect was about 0.07, so this does not show "no effect".
2. **The small gain is consistent across seeds:** +0.040 / +0.028 / +0.033 (mean +0.033, SD 0.006). It is positive in all 3 seeds (descriptive).
3. **exp09's +0.068 splits roughly into two parts:**
   - **+0.025 from more training:** M1-ctrl vs exp09's M1, 12 vs 2 discordant questions, p = 0.013 (descriptive).
   - **about +0.04 from conditioning:** not significant.
   - So H6 ("iteration beats one-shot") is **not confirmed** once the confound is removed.
4. **The advantage is concentrated in 4-hop chains (exploratory, uncorrected).** 4-hop: **0.578 vs 0.391** (20 vs 8 discordant questions, p = 0.036, n = 64). 2-hop: 0.823 vs 0.813. 3-hop: 0.714 vs 0.699.
5. **Secondary holds: the pure conditioning change is rank 2.** D = M1-iter − M1-ctrl, grafted onto M1-ctrl at rank 2, recovers **86.8% [77.7, 95.8]** of the gain on test (iterative inference: 0.4075 → 0.7025, full 0.7475).
   - The norm-matched random rank-2 control recovers 7% on dev300, against 85% for the real rank 2.
   - **What this change does:** it is the skill of *reading* a conditioned query. Without it, M1-ctrl collapses when run iteratively (0.41). The change is functionally rank 2, even though its benefit over a matched one-shot ranker is small overall.

## 2×2: model × inference (test recall@5, seed 0)

| | One-shot inference | Iterative inference |
|---|---|---|
| **M1-iter** (trained with conditioning) | 0.685 | **0.7475** |
| **M1-ctrl** (same pairs, no conditioning) | **0.7075** | 0.4075 |

Each model is best in the inference mode it was trained for. Conditioning hurts one-shot use a little (0.685 vs 0.7075).

## Recall@k (test, seed 0)

| k | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|
| M1-iter, iterative | **0.313** | **0.530** | **0.675** | **0.748** | 0.775 | **0.848** | 0.898 | 0.958 |
| M1-ctrl, one-shot | 0.283 | 0.505 | 0.595 | 0.708 | **0.785** | 0.845 | **0.913** | **0.980** |
| M1-ctrl + rank-2 D, iterative | 0.298 | 0.500 | 0.648 | 0.703 | 0.753 | 0.848 | 0.880 | 0.960 |

The pattern from exp09/exp10 holds with matched data: iteration helps at the top (k ≤ 5, mostly on 4-hop questions) and loses in the tail (k ≥ 10).

## What this changes in the claim ledger

- **H6 (iteration beats one-shot): from C? to N at matched data.** There is a small, seed-consistent effect (+0.03), and an exploratory 4-hop effect that must be re-tested before anyone claims it.
- **H9 (the iteration change is low-rank): confirmed in the de-confounded form.** Both D_it (exp11, 84.7%) and the pure conditioning change D (exp12, 86.8%) are about rank 2.
- **For the thesis, the honest framing:**
  - *Practical:* in-domain one-shot ranking plus adaptive stopping is the efficient choice. Iteration costs 3.7× the passes for about +0.03 recall@5 overall.
  - *Research:* the ability to use a conditioned query is a separable, rank-2 change. Its value shows up on long chains (4-hop), which motivates a cascade: switch it on only for hard or long questions.

## Deviations

1. **Input staging.** This PC's Google Drive belongs to another account, so the inputs come from the Colab account's Drive (A1, sha-checked) and from the official MuSiQue release (sha-checked). No model was trained before the fix. The first train-subset hash had been computed on a Windows CRLF copy; the LF copy was verified byte-identical.
2. **Results transfer.** The Colab Drive is not synced here. `results.json` and the per-question hits were copied from the notebook output, and the copy was checked: recall@5 recomputed from the hits matches exactly, and so does the McNemar test.

## Files

- `PROTOCOL.md`, `train.py` (`--mode ctrl`, `--seed`), `evaluate.py`, `analyze.py`, `make_notebook.py`, `exp12_colab.ipynb`
- `results/results.json`, `results/hits_compact.json`
