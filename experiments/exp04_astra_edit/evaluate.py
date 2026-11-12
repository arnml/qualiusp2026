"""Evaluate an exp04 run's final edit on held-out questions (never seen by the editor).

  uv run python evaluate.py --run s1_C_B5                 # 300 dev questions (stage gates)
  uv run python evaluate.py --run s1_C_B5 --eval-n 1000   # full dev
  uv run python evaluate.py --run s3_C_B50 --test         # + the 400 test questions (stage 3 only)

Reports both-gold@3, encoder GFLOPs (cost()), and the best baseline at the closest FLOPs
from results/baselines_dev300.json. Writes runs/<run>/eval.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import editlib as el
from baselines import MODEL, bg3, flatten, score
from data import load_eval

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    run = HERE / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text())
    st = el.EditState.load(run / "work" / "edited", cfg["track"])
    errs = st.check(el.load_base(MODEL))
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    base = el.load_base(MODEL)
    model = el.build_model(MODEL, st, base)
    c = st.cost()
    full = el.EditState("C").cost()["encoder_gflops_per_question"]
    out = {"run": args.run, "valid": not errs, "errors": errs, "cost": c,
           "flops_fraction": c["encoder_gflops_per_question"] / full}

    dev = load_eval("dev")[: args.eval_n]
    q, p, owner = flatten(dev)
    out[f"dev{args.eval_n}_bg3"] = bg3(score(model, tok, q, p), dev, owner)
    if args.test:
        test = load_eval("test")
        q, p, owner = flatten(test)
        out["test_bg3"] = bg3(score(model, tok, q, p), test, owner)

    bfile = HERE / "results" / "baselines_dev300.json"
    if bfile.exists() and args.eval_n == 300:
        rows = [r for r in json.loads(bfile.read_text()) if r["method"] != "layerdrop"]
        g = c["encoder_gflops_per_question"]
        near = min({r["frac"] for r in rows if r["method"] != "A1_unedited"},
                   key=lambda f: abs(f * full - g))
        out["closest_baselines"] = {r["method"]: {"gflops": r["gflops"], "dev300_bg3": r["dev_bg3"]}
                                    for r in rows if r["frac"] in (near, 1.0)}
    (run / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
