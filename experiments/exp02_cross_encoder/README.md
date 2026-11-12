# exp02: small learned cross-encoder selector (option 2)

Scores each (question, paragraph) pair with `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M params) and keeps either the top-k paragraphs or those above a threshold. It uses the same 400 test / 1000 dev questions as exp01; the IDs are in `split_ids.json`, so the questions are fixed regardless of dataset order.

| Arm | What | Where |
|---|---|---|
| A0 | zero-shot (off-the-shelf MS MARCO reranker) | CPU ok (~10 min scoring) |
| A2 | fine-tuned on N = 50 / 200 train questions (fair baseline for exp03 LLM editing) | Colab T4 (CPU possible) |
| A1 | fine-tuned on all ~90k train questions, 1 epoch, ~900k pairs | Colab T4 |

## Run

**Colab (T4):** open `exp02_colab.ipynb` and *Run all*.
- Results go to Google Drive (`MyDrive/qualiusp2026/exp02_results`) during training.
- Checkpoints are written every 2000 steps and resume automatically after a disconnect.
- The last cell downloads a zip of the small files; unzip it into `results/`.

**Local:**

```bash
uv sync
uv run python score.py --model zeroshot
uv run python train.py --n-train 200 --epochs 10 --tag n200
uv run python score.py --model n200
uv run python analyze.py
```

The notebook is generated from the .py files by `uv run python make_notebook.py`. Edit the .py files, not the notebook.

## Output

- `results/scores_<model>.json`: one logit per paragraph, for dev and test.
- `results/summary.csv`: test metrics per scorer for top-1…5, plus thresholds chosen on dev for 0.90 / 0.95 full recall.
- `results/dev_thresholds_<model>.csv`: the full threshold sweep on dev.
- Model weights stay on Google Drive and are not committed.

## Results (2026-11-12, Colab T4; test 400, thresholds chosen on dev)

The full fine-tune (A1) took 28,115 steps, ~50 min on the T4. Colab and local CPU zero-shot scores agree (corr 1.000000, identical top-2 on every question).

| Arm | Rule | Both-gold | Precision | Paragraphs | Tokens | vs k=10 |
|---|---|---|---|---|---|---|
| — | k=10 (all) | 1.000 | 0.20 | 10.0 | 1442 | — |
| exp01 | Noise-Gate tau=0.20 | 0.928 | 0.25 | 8.4 | 1221 | −15% |
| exp01 | hop gate | 0.918 | 0.29 | 7.6 | 1089 | −24% |
| A0 zero-shot | threshold @0.90 | 0.915 | 0.27 | 8.2 | 1143 | −21% |
| A2 n50 | threshold @0.90 | 0.932 | 0.38 | 6.4 | 888 | −38% |
| A2 n200 | threshold @0.90 | 0.915 | 0.46 | 5.3 | 733 | −49% |
| **A1 full** | **top-2** | **0.878** | 0.94 | 2.0 | **324** | **−78%** |
| **A1 full** | **threshold @0.90 (t=0.07)** | **0.915** | 0.83 | 2.6 | **391** | **−73%** |
| **A1 full** | **threshold @0.95 (t=0.02)** | **0.968** | 0.74 | 3.1 | **461** | **−68%** |
| — | gold paragraphs only (oracle) | 1.000 | 1.00 | 2.0 | 323 | −78% |

**Readout**
- At the same both-gold level (~0.92), the fine-tuned 22.7M selector sends **391 tokens instead of 1089** (hop gate) or 1221 (Noise-Gate). Its top-2 uses the same tokens as the gold oracle.
- Literature, large encoders (dev, exactly both gold in the top 2): SAE 92.0%, Beam Retrieval 97.5%. A1 top-2 reaches 87.8% (test) with a 22.7M model.
- The data-matched baselines leave a large gap: n200 needs 733 tokens for 0.915, against A1's 391. That's the room exp03 (LLM weight editing) tries to close.
- Selector cost, rough estimate: 10 pairs × ~160 tokens × 2 × 22.7M ≈ 0.07 TFLOPs per question. Saving ~830 tokens against Noise-Gate on an 8B generator ≈ 13 TFLOPs, so the selector costs ~1/180 of what it saves.
### Answer quality with the paper's generator (2026-11-12)

Setup: `generate.py`, gpt-5.4-mini via the OpenAI API with default params, the paper's exact prompt and F1, the same 400 test questions. 2,000 calls, no failures.
- Tokens are the API-reported prompt tokens; USD uses the paper's prices.
- ΔF1 is paired against today's k=10 run (same day, same model), with a 95% bootstrap CI over the 400 questions (5,000 resamples).

| Context | F1 | EM | Tokens | USD / query | ΔF1 vs k=10 [95% CI] |
|---|---|---|---|---|---|
| k=10, all paragraphs (today) | 0.782 | 0.620 | 1448 | 2.95e-4 | — |
| *Paper k=10 (BRACIS)* | *0.792* | — | *1452.5* | *2.95e-4* | *same setup, earlier run* |
| *Paper Noise-Gate tau=0.30* | *0.764* | — | *1065.3* | *2.17e-4* | — |
| **A1 threshold t=0.02** | **0.796** | 0.630 | **467 (−68%)** | 0.99e-4 | **+0.014 [−0.013, +0.043]** |
| A1 threshold t=0.07 | 0.778 | 0.623 | 397 (−73%) | 0.85e-4 | −0.004 [−0.033, +0.026] |
| A1 top-2 | 0.780 | 0.628 | 330 (−77%) | 0.71e-4 | −0.001 [−0.033, +0.030] |
| Oracle gold paragraphs | 0.822 | 0.670 | 329 | 0.71e-4 | +0.040 [+0.017, +0.064] |

**Readout**
- All three A1 rules are **statistically indistinguishable from k=10**, and all are inside the paper's ±0.05 band, using 68–77% fewer tokens and costing a third to a quarter as much per query. In the paper, Noise-Gate saved 27% tokens at −0.028 F1.
- The gold-only oracle is *better* than k=10 (+0.040, CI excludes 0). Distractors actively hurt the generator, so a good selector can improve answers, not only save tokens.
- Today's k=10 (0.782) is within 0.01 of the paper's run (0.792), so the setup reproduces.
- Remaining gap to the oracle: ~0.026 F1 (t=0.02) at +138 tokens. That's room for sentence-level selection or a better selector.

## Reference numbers (exp01, test 400)

| Method | Full recall | Tokens |
|---|---|---|
| k=10 | 1.000 | 1442 |
| Noise-Gate tau=0.20 | 0.928 | 1221 |
| Hop-gate | 0.918 | 1089 |
| Gold paragraphs only | 1.000 | 323 |

Literature ceiling (large encoders, picking exactly both gold paragraphs, dev): SAE 92.0%, Beam Retrieval 97.5%.
