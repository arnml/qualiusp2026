#!/usr/bin/env bash
set -e
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
uv run python rank_scores.py --arm M1
uv run python rank_scores.py --arm M1-iter
echo RANK_DONE
