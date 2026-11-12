"""Paired McNemar (exact) on dev questions: an editor run vs Wanda at a given FLOPs fraction.

  uv run python mcnemar.py --run s2_C_B10 --frac 0.7276 --eval-n 1000
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import editlib as el
from baselines import MODEL, flatten, greedy_state, importances, score
from data import gold_mask, load_eval
from harness import build_practice

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--eval-n", type=int, default=1000)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    base = el.load_base(MODEL)
    dev = load_eval("dev")[: args.eval_n]
    q, p, owner = flatten(dev)

    def per_q(st):
        s = score(el.build_model(MODEL, st, base), tok, q, p)
        out = []
        for k, r in enumerate(dev):
            g = np.array(gold_mask(r))
            out.append(g[np.argsort(-s[owner == k])[:3]].sum() == g.sum())
        return np.array(out)

    run = HERE / "runs" / args.run
    ed = per_q(el.EditState.load(run / "work" / "edited", "C"))
    imp = importances(el.build_model(MODEL, el.EditState("C"), base), tok, build_practice("heldval", 200))
    wa = per_q(greedy_state(*imp["wanda"], args.frac))
    b, c = int((ed & ~wa).sum()), int((~ed & wa).sum())
    n = b + c
    pval = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    res = {"run": args.run, "frac": args.frac, "editor": float(ed.mean()), "wanda": float(wa.mean()),
           "editor_only": b, "wanda_only": c, "mcnemar_exact_p": pval}
    (run / f"mcnemar_dev{args.eval_n}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
