"""Re-run exp09 inference (M1 / M1-iter) and record selection scores (PROTOCOL.md). Resumable per question.

  uv run python rank_scores.py --arm M1-iter --splits test,dev300,holdout
Writes results/rank_<arm>_<split>.jsonl: {"id", "order", "sel": score that placed each position, "step_passes": [...]}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from data import encode, load, paragraphs_of, query
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_iterative_rerank" / "models"
MODELS = {"M1": E9 / "m1" / "model", "M1-iter": E9 / "m1_iter" / "model"}
STEPS = 4
torch.set_num_threads(8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(MODELS))
    ap.add_argument("--splits", default="test,dev300,holdout")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(MODELS[args.arm]))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELS[args.arm])).eval()
    iterative = args.arm == "M1-iter"

    def score(q, ps):
        return model(**encode(tok, [q] * len(ps), ps)).logits.squeeze(-1).reshape(-1).tolist()

    for split in args.splits.split(","):
        path = HERE / "results" / f"rank_{args.arm}_{split}.jsonl"
        done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        rows = [r for r in load(split) if r["id"] not in done]
        t0 = time.time()
        for n, r in enumerate(rows):
            _, paras = paragraphs_of(r)
            rest, order, sel, step_passes = list(range(len(paras))), [], [], []
            for _ in range(STEPS if iterative else 1):
                s = score(query(r["question"], [paras[i] for i in order]), [paras[i] for i in rest])
                step_passes.append(len(rest))
                ranked = sorted(zip(s, rest), key=lambda z: -z[0])
                if not iterative:
                    order, sel = [i for _, i in ranked], [v for v, _ in ranked]
                    break
                order.append(ranked[0][1])
                sel.append(ranked[0][0])
                last = ranked[1:]
                rest = [i for _, i in last]
            if iterative:  # tail ordered by the last step's scores (exp09)
                order += [i for _, i in last]
                sel += [v for v, _ in last]
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"id": r["id"], "order": order, "sel": [round(v, 5) for v in sel], "step_passes": step_passes}
                    )
                    + "\n"
                )
            if n % 100 == 0:
                print(args.arm, split, n, len(rows), round(time.time() - t0), "s", flush=True)
        print(args.arm, split, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
