"""exp11: is the iteration skill low-rank and localized? Graft parts of a fine-tuning delta onto its base (PROTOCOL.md).

  uv run python graft.py --split dev300                 # all pre-registered configurations (resumable)
  uv run python graft.py --split test --only it_r4      # confirmation of one configuration

Families:
  it_*  : delta D_it  = M1-iter - M1, grafted onto M1, ITERATIVE inference (exp09: 4 greedy steps)
  dom_* : delta D_dom = M1 - A1,      grafted onto A1, ONE-SHOT inference
No data is used to choose a graft (SVD, random, fixed layer groups), so dev300 is a fair sweep set.
Writes results/grafts_<split>.jsonl: one line per config with per-question hit@5 / hit@10 and recall@k.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from data import encode, gold_mask, load, paragraphs_of, query
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_iterative_rerank" / "models"
DIRS = {
    "A1": HERE.parent / "exp04_astra_edit" / "models" / "a1",
    "M1": E9 / "m1" / "model",
    "M1-iter": E9 / "m1_iter" / "model",
}
STEPS, KS = 4, (2, 3, 4, 5, 6, 8, 10, 15)
torch.set_num_threads(8)


def state(name):
    return {k: v.float() for k, v in load_file(str(DIRS[name] / "model.safetensors")).items()}


def group_of(n):
    if n.startswith("bert.embeddings."):
        return "emb"
    if n.startswith(("bert.pooler.", "classifier.")):
        return "head"
    return "L" + re.match(r"bert\.encoder\.layer\.(\d+)\.", n).group(1)


def lowrank(base, d, r, rand=False, seed=0):
    """Rank-r truncated SVD of every 2-D delta (1-D deltas in full), as exp06.
    rand: a random rank-r matrix with the same Frobenius norm as the rank-r SVD part (control)."""
    g, out, n_par = torch.Generator().manual_seed(seed), {}, 0
    for n in base:
        if d[n].dim() == 2 and min(d[n].shape) > r:
            U, S, Vh = torch.linalg.svd(d[n], full_matrices=False)
            part = (U[:, :r] * S[:r]) @ Vh[:r]
            if rand:
                R = torch.randn(d[n].shape[0], r, generator=g) @ torch.randn(r, d[n].shape[1], generator=g)
                part = R * (part.norm() / R.norm())
            out[n] = base[n] + part
            n_par += r * sum(d[n].shape)
        else:
            out[n] = base[n] + d[n]
            n_par += d[n].numel()
    return out, n_par


def layer_only(base, d, layer):
    out, n_par = {}, 0
    for n in base:
        use = group_of(n) == layer
        out[n] = base[n] + d[n] if use else base[n].clone()
        n_par += d[n].numel() if use else 0
    return out, n_par


def configs(S):
    fam = {"it": ("M1", "M1-iter", True), "dom": ("A1", "M1", False)}
    c = {}
    for f, (b, t, iterative) in fam.items():
        base = S[b]
        d = {n: S[t][n] - base[n] for n in base}
        c[f"{f}_base"] = (lambda base=base: ({n: v.clone() for n, v in base.items()}, 0), iterative)
        c[f"{f}_full"] = (lambda t=t: ({n: v.clone() for n, v in S[t].items()}, -1), iterative)
        for r in (1, 2, 4, 8, 16):
            c[f"{f}_r{r}"] = (lambda base=base, d=d, r=r: lowrank(base, d, r), iterative)
        c[f"{f}_rand_r4"] = (lambda base=base, d=d: lowrank(base, d, 4, rand=True), iterative)
        for l in range(6):
            c[f"{f}_L{l}"] = (lambda base=base, d=d, l=l: layer_only(base, d, f"L{l}"), iterative)
    return c


def build(sd):
    m = AutoModelForSequenceClassification.from_pretrained(str(DIRS["A1"]))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all("position_ids" in k for k in missing), (missing, unexpected)
    return m.eval()


@torch.no_grad()
def rank_all(model, tok, rows, iterative):
    def score(q, ps):
        return model(**encode(tok, [q] * len(ps), ps)).logits.squeeze(-1).reshape(-1).tolist()

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
    ap.add_argument("--smoke", type=int, default=0, help="first N questions only, written to a separate smoke file")
    args = ap.parse_args()
    S = {k: state(k) for k in DIRS}
    tok = AutoTokenizer.from_pretrained(str(DIRS["A1"]))
    rows = load(args.split)[: args.smoke or None]
    out = HERE / "results" / f"grafts_{args.split}{f'_smoke{args.smoke}' if args.smoke else ''}.jsonl"
    done = {json.loads(x)["config"] for x in out.read_text().splitlines()} if out.exists() else set()
    for name, (make, iterative) in configs(S).items():
        if (args.only and name not in args.only) or name in done:
            continue
        t0 = time.time()
        sd, n_par = make()
        orders = rank_all(build(sd), tok, rows, iterative)
        hits = {
            k: [int(np.array(gold_mask(r), bool)[o[:k]].sum() == sum(gold_mask(r))) for r, o in zip(rows, orders)]
            for k in KS
        }
        rec = {
            "config": name,
            "iterative": iterative,
            "params_grafted": int(n_par),
            "recall": {k: round(float(np.mean(v)), 4) for k, v in hits.items()},
            "hit5": hits[5],
            "hit10": hits[10],
            "ids": [r["id"] for r in rows],
            "seconds": round(time.time() - t0),
        }
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, rec["recall"][5], rec["recall"][10], rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
