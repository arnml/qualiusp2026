"""exp12 evaluation (PROTOCOL.md): GPU models m1_iter_s0 / m1_ctrl (+ seeds 1-2), 2x2 model x inference,
and grafts of D = m1_iter_s0 - m1_ctrl onto m1_ctrl. Run from exp12_colab.ipynb.

  python evaluate.py --split test --only iter_iter ctrl_oneshot iter_oneshot ctrl_iter iter_iter_s1 ...
  python evaluate.py --split dev300 --only ctrl_iter iter_iter g_r1 g_r2 g_r4 g_r8 g_rand_r2
Writes results/eval_<split>.jsonl: one line per config, with per-question orders (so any recall@k can be recomputed).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from data import encode, gold_mask, load, paragraphs_of, query
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_iterative_rerank" / "models"
MODELS = Path(os.environ.get("MODELS_DIR", HERE / "models"))
DIRS = {
    "A1": Path(os.environ.get("A1_DIR", HERE.parent / "exp04_astra_edit" / "models" / "a1")),
    "M1": Path(os.environ.get("M1_DIR", E9 / "m1" / "model")),
    "M1-iter": Path(os.environ.get("M1_ITER_DIR", E9 / "m1_iter" / "model")),
    "M1-ctrl": MODELS / "m1_ctrl" / "model",
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 4
torch.set_num_threads(8)


def state(name):
    return {k: v.float() for k, v in load_file(str(DIRS[name] / "model.safetensors")).items()}


def lowrank(base, d, r, rand=False, seed=0):
    """As exp06/exp11: rank-r truncated SVD of each 2-D delta, 1-D deltas in full; rand = norm-matched random rank r."""
    g, out = torch.Generator().manual_seed(seed), {}
    for n in base:
        if d[n].dim() == 2 and min(d[n].shape) > r:
            U, S, Vh = torch.linalg.svd(d[n], full_matrices=False)
            part = (U[:, :r] * S[:r]) @ Vh[:r]
            if rand:
                R = torch.randn(d[n].shape[0], r, generator=g) @ torch.randn(r, d[n].shape[1], generator=g)
                part = R * (part.norm() / R.norm())
            out[n] = base[n] + part
        else:
            out[n] = base[n] + d[n]
    return out


def tagged(tag):
    return {k: v.float() for k, v in load_file(str(MODELS / tag / "model" / "model.safetensors")).items()}


def configs():
    """PROTOCOL (after the Codex pre-run review): primary pair = GPU M1-iter seed 0 (m1_iter_s0) vs M1-ctrl seed 0
    (m1_ctrl); D = m1_iter_s0 - m1_ctrl; 2x2 model x inference; seeds 1-2 in their native inference."""
    it, ct = tagged("m1_iter_s0"), tagged("m1_ctrl")
    d = {n: it[n] - ct[n] for n in ct}
    c = {
        "iter_iter": (lambda: it, True),
        "ctrl_oneshot": (lambda: ct, False),  # primary system comparison
        "iter_oneshot": (lambda: it, False),
        "ctrl_iter": (lambda: ct, True),
    }  # 2x2 (descriptive); ctrl_iter = graft base
    for r in (1, 2, 4, 8):
        c[f"g_r{r}"] = (lambda r=r: lowrank(ct, d, r), True)
    c["g_rand_r2"] = (lambda: lowrank(ct, d, 2, rand=True), True)
    for s in (1, 2):
        c[f"iter_iter_s{s}"] = (lambda s=s: tagged(f"m1_iter_s{s}"), True)
        c[f"ctrl_oneshot_s{s}"] = (lambda s=s: tagged(f"m1_ctrl_s{s}"), False)
    return c


def build(sd):
    m = AutoModelForSequenceClassification.from_pretrained(str(DIRS["A1"]))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all("position_ids" in k for k in missing), (missing, unexpected)
    return m.eval().to(DEV)


@torch.no_grad()
def rank_all(model, tok, rows, iterative):
    """exp09 inference: one-shot sort, or 4 greedy steps with the tail ordered by the step-3 scores."""

    def score(q, ps):
        return model(**encode(tok, [q] * len(ps), ps).to(DEV)).logits.float().cpu().squeeze(-1).reshape(-1).tolist()

    orders = []
    for r in rows:
        _, paras = paragraphs_of(r)
        rest, order = list(range(len(paras))), []
        for _ in range(STEPS if iterative else 1):
            s = score(query(r["question"], [paras[i] for i in order]), [paras[i] for i in rest])
            ranked = [i for _, i in sorted(zip(s, rest), key=lambda z: -z[0])]
            if not iterative:
                order = ranked
                break
            order.append(ranked[0])
            rest = ranked[1:]
        orders.append(order + (rest if iterative else []))
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev300", choices=("dev300", "test"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(DIRS["A1"]))
    rows = load(args.split)[: args.smoke or None]
    ids = json.loads((HERE / "split_ids.json").read_text())
    assert not set(ids["dev300"]) & set(ids["test"]) and not set(ids["train"]) & set(ids["dev300"] + ids["test"])
    out = (
        Path(os.environ.get("RESULTS_DIR", HERE / "results"))
        / f"eval_{args.split}{f'_smoke{args.smoke}' if args.smoke else ''}.jsonl"
    )
    done = {json.loads(x)["config"] for x in out.read_text().splitlines()} if out.exists() else set()
    for name, (make, iterative) in configs().items():
        if (args.only and name not in args.only) or name in done:
            continue
        t0 = time.time()
        orders = rank_all(build(make()), tok, rows, iterative)
        hit5 = [int(np.array(gold_mask(r), bool)[o[:5]].sum() == sum(gold_mask(r))) for r, o in zip(rows, orders)]
        with out.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "config": name,
                        "iterative": iterative,
                        "ids": [r["id"] for r in rows],
                        "orders": orders,
                        "recall5": round(float(np.mean(hit5)), 4),
                        "seconds": round(time.time() - t0),
                    }
                )
                + "\n"
            )
        print(name, round(float(np.mean(hit5)), 4), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
