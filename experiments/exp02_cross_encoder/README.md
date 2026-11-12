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

## Reference numbers (exp01, test 400)

| Method | Full recall | Tokens |
|---|---|---|
| k=10 | 1.000 | 1442 |
| Noise-Gate tau=0.20 | 0.928 | 1221 |
| Hop-gate | 0.918 | 1089 |
| Gold paragraphs only | 1.000 | 323 |

Literature ceiling (large encoders, picking exactly both gold paragraphs, dev): SAE 92.0%, Beam Retrieval 97.5%.
