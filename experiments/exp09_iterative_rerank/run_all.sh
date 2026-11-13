#!/usr/bin/env bash
# exp09 pipeline (n_questions = 1000 per the budget rule; see results/budget.json)
set -e
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
uv run python rank.py --arm A1
uv run python train.py --mode step0 --tag m1 --n-questions 1000
uv run python rank.py --arm M1
uv run python train.py --mode iter --tag m1_iter --n-questions 1000
uv run python rank.py --arm M1-iter
uv run python analyze.py > results/analyze_partial_log.txt
uv run python rank.py --arm A1-iter
uv run python analyze.py
echo PIPELINE_DONE
