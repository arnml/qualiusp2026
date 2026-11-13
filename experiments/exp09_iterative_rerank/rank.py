"""Rank every paragraph of dev300 / test with one arm (PROTOCOL.md). Resumable per question.

  uv run python rank.py --arm A1 | A1-iter | M1 | M1-iter
Writes results/rank_<arm>_<split>.jsonl: {"id", "order": [paragraph positions, best first], "passes"}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import encode, load, paragraphs_of, query

HERE = Path(__file__).resolve().parent
MODELS = {"A1": HERE.parent / "exp04_astra_edit" / "models" / "a1", "M1": HERE / "models" / "m1" / "model",
          "M1-iter": HERE / "models" / "m1_iter" / "model"}
MODELS["A1-iter"] = MODELS["A1"]
STEPS = 4
torch.set_num_threads(8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(MODELS))
    ap.add_argument("--splits", default="test,dev300")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(MODELS[args.arm]))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELS[args.arm])).eval()
    iterative = args.arm.endswith("-iter")

    def score(q, ps):
        enc = encode(tok, [q] * len(ps), ps)
        return model(**enc).logits.squeeze(-1).reshape(-1).tolist()

    (HERE / "results").mkdir(exist_ok=True)
    for split in args.splits.split(","):
        path = HERE / "results" / f"rank_{args.arm}_{split}.jsonl"
        done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        rows = [r for r in load(split) if r["id"] not in done]
        t0 = time.time()
        for n, r in enumerate(rows):
            _, paras = paragraphs_of(r)
            rest, sel, passes = list(range(len(paras))), [], 0
            for _ in range(STEPS if iterative else 1):
                s = score(query(r["question"], [paras[i] for i in sel]), [paras[i] for i in rest])
                passes += len(rest)
                ranked = [i for _, i in sorted(zip(s, rest), key=lambda z: -z[0])]
                if not iterative:
                    sel = ranked; break
                sel.append(ranked[0]); rest = ranked[1:]
            order = sel if not iterative else sel + rest  # rest is ordered by the last step's scores
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": r["id"], "order": order, "passes": passes}) + "\n")
            if n % 50 == 0:
                print(args.arm, split, n, len(rows), round(time.time() - t0), "s", flush=True)
        print(args.arm, split, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
