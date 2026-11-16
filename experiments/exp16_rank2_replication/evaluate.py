"""exp16 evaluation (PROTOCOL.md): per dataset, the 2x2 (model x inference) and grafts of D = iter - ctrl onto ctrl.

  python evaluate.py --ds tw --split tw_tune --only ctrl_iter iter_iter g_r1 g_r2 g_r4 g_r8 g_rand_r2
Writes RESULTS_DIR/eval_<split>.jsonl (one line per config, with orders and the model hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from data import encode, load, query
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
A1 = Path(os.environ.get("A1_DIR", HERE.parent / "exp04_astra_edit" / "models" / "a1"))
MODELS = Path(os.environ.get("MODELS_DIR", HERE / "models"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 4


def sd(tag):
    return {k: v.float() for k, v in load_file(str(MODELS / tag / "model" / "model.safetensors")).items()}


def lowrank(base, d, r, rand=False, seed=0):
    """exp11/exp12: rank-r truncated SVD of each 2-D delta (1-D deltas in full); rand = norm-matched random rank r."""
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


def configs(ds):
    it, ct, c1 = sd(f"{ds}_iter"), sd(f"{ds}_ctrl"), sd(f"{ds}_ctrl_s1")
    d = {n: it[n] - ct[n] for n in ct}
    noise = {n: c1[n] - ct[n] for n in ct}  # seed-noise null: same training, different seed
    c = {
        "iter_iter": (lambda: it, True),
        "ctrl_oneshot": (lambda: ct, False),
        "iter_oneshot": (lambda: it, False),
        "ctrl_iter": (lambda: ct, True),
    }
    for r in (1, 2, 4, 8):
        c[f"g_r{r}"] = (lambda r=r: lowrank(ct, d, r), True)
    c["g_rand_r2"] = (lambda: lowrank(ct, d, 2, rand=True), True)
    c["g_noise_r2"] = (lambda: lowrank(ct, noise, 2), True)
    return c


@torch.no_grad()
def rank_all(model, tok, rows, iterative):
    def score(q, ps):
        return model(**encode(tok, [q] * len(ps), ps).to(DEV)).logits.float().cpu().squeeze(-1).reshape(-1).tolist()

    orders = []
    for r in rows:
        paras = r["paras"]
        rest, order, qlen = list(range(len(paras))), [], []
        for _ in range(STEPS if iterative else 1):
            if not rest:
                break
            q = query(r["question"], [paras[i] for i in order])
            qlen.append(len(tok(q)["input_ids"]))
            s = score(q, [paras[i] for i in rest])
            ranked = [i for _, i in sorted(zip(s, rest), key=lambda z: -z[0])]
            if not iterative:
                order = ranked
                rest = []
                break
            order.append(ranked[0])
            rest = ranked[1:]
        order = order + rest
        assert sorted(order) == list(range(len(paras)))
        orders.append((order, max(qlen)))
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=("hp", "tw"), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--only", nargs="+", required=True)
    args = ap.parse_args()
    assert DEV == "cuda" or os.environ.get("ALLOW_CPU"), "PROTOCOL: Colab GPU"
    assert args.split.startswith(args.ds)
    tok = AutoTokenizer.from_pretrained(str(A1))
    rows = load(args.split)
    out = Path(os.environ.get("RESULTS_DIR", HERE / "results")) / f"eval_{args.split}.jsonl"
    done = {json.loads(x)["config"] for x in out.read_text().splitlines()} if out.exists() else set()
    hashes = {
        t: hashlib.sha256((MODELS / t / "model" / "model.safetensors").read_bytes()).hexdigest()[:16]
        for t in (f"{args.ds}_iter", f"{args.ds}_ctrl", f"{args.ds}_ctrl_s1")
    }
    cf = configs(args.ds)
    for name in args.only:
        if name in done:
            continue
        make, iterative = cf[name]
        t0 = time.time()
        m = AutoModelForSequenceClassification.from_pretrained(str(A1))
        m.load_state_dict(make(), strict=False)
        m = m.eval().to(DEV)
        res = rank_all(m, tok, rows, iterative)
        orders = [o for o, _ in res]
        with out.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "config": name,
                        "iterative": iterative,
                        "ids": [r["id"] for r in rows],
                        "orders": orders,
                        "max_query_tokens": [q for _, q in res],
                        "model_sha": hashes,
                        "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                    }
                )
                + "\n"
            )
        print(args.split, name, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
