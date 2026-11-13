"""exp06: how small is the multi-hop skill? Graft parts of the fine-tuning delta (A1 - A0) into A0.

  uv run python graft.py --eval-n 1000            # all pre-registered configurations
  uv run python graft.py --only lowrank_r4        # one configuration

No data is used to choose what is grafted (magnitude, SVD, random, or fixed component groups), so
dev questions are a fair evaluation set. Results: results/grafts_dev<N>.jsonl (one line per config,
appended, so an interrupted run resumes).
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch

from common import A0_DIR, A1_DIR, HERE, build, metrics, prep, score, state

ASTRA_CAP = 108_000


def group_of(name: str) -> str:
    if name.startswith("bert.embeddings."):
        return "emb"
    if name.startswith(("bert.pooler.", "classifier.")):
        return "head"
    m = re.match(r"bert\.encoder\.layer\.(\d+)\.(.*)", name)
    if not m:
        raise ValueError(name)
    l, rest = m.group(1), m.group(2)
    part = "attn" if rest.startswith("attention.") else "mlp"
    return f"L{l}.{part}"


def in_astra_scope(name: str) -> bool:
    return name.startswith(("bert.encoder.layer.", "bert.pooler.dense."))


def sparse(a0, d, k: int, scope=lambda n: True, rand: bool = False, seed: int = 0):
    """Top-k |delta| entries (or k random entries) within scope, grafted onto A0."""
    names = [n for n in d if scope(n)]
    flat = torch.cat([d[n].flatten() for n in names])
    if rand:
        idx = torch.from_numpy(np.random.default_rng(seed).choice(flat.numel(), k, replace=False))
    else:
        idx = torch.topk(flat.abs(), k).indices
    mask = torch.zeros_like(flat, dtype=torch.bool); mask[idx] = True
    out, off = {}, 0
    for n in a0:
        if n in names:
            m = mask[off:off + d[n].numel()].view_as(d[n]); off += d[n].numel()
            out[n] = a0[n] + d[n] * m
        else:
            out[n] = a0[n].clone()
    return out, k


def lowrank(a0, d, r: int):
    """Rank-r truncated SVD of every 2-D delta (PHLoRA-style); 1-D deltas grafted in full."""
    out, n_par = {}, 0
    for n in a0:
        if d[n].dim() == 2 and min(d[n].shape) > r:
            U, S, Vh = torch.linalg.svd(d[n], full_matrices=False)
            out[n] = a0[n] + (U[:, :r] * S[:r]) @ Vh[:r]
            n_par += r * sum(d[n].shape)
        else:
            out[n] = a0[n] + d[n]; n_par += d[n].numel()
    return out, n_par


def groups(a0, d, keep: set[str] | None = None, drop: set[str] | None = None, only_1d: bool = False):
    """Graft the delta of selected parameter groups (or everything except `drop`)."""
    out, n_par = {}, 0
    for n in a0:
        g = group_of(n)
        use = (g in keep) if keep is not None else (g not in drop)
        if only_1d:
            use = d[n].dim() == 1 and not n.startswith("bert.embeddings.position")
        out[n] = a0[n] + d[n] if use else a0[n].clone()
        n_par += d[n].numel() if use else 0
    return out, n_par


def configs(a0, a1, d):
    total = sum(v.numel() for v in d.values())
    L = [f"L{i}" for i in range(6)]
    c = {"A0": lambda: ({n: v.clone() for n, v in a0.items()}, 0),
         "A1": lambda: ({n: v.clone() for n, v in a1.items()}, total)}
    for f in (0.1, 0.01, 0.001, 0.0001):
        c[f"sparse_top_{f:g}"] = lambda f=f: sparse(a0, d, int(f * total))
    c["sparse_random_0.01"] = lambda: sparse(a0, d, int(0.01 * total), rand=True)
    c["astra_scope_top_108k"] = lambda: sparse(a0, d, ASTRA_CAP, in_astra_scope)
    c["astra_scope_random_108k"] = lambda: sparse(a0, d, ASTRA_CAP, in_astra_scope, rand=True)
    for r in (1, 2, 4, 8, 16, 32):
        c[f"lowrank_r{r}"] = lambda r=r: lowrank(a0, d, r)
    c["only_1d"] = lambda: groups(a0, d, only_1d=True)
    c["only_emb"] = lambda: groups(a0, d, keep={"emb"})
    c["only_head"] = lambda: groups(a0, d, keep={"head"})
    c["only_attn_all"] = lambda: groups(a0, d, keep={f"{l}.attn" for l in L})
    c["only_mlp_all"] = lambda: groups(a0, d, keep={f"{l}.mlp" for l in L})
    for l in L:
        c[f"only_{l}"] = lambda l=l: groups(a0, d, keep={f"{l}.attn", f"{l}.mlp"})
    c["without_emb"] = lambda: groups(a0, d, drop={"emb"})
    c["without_head"] = lambda: groups(a0, d, drop={"head"})
    for l in L:
        c[f"without_{l}"] = lambda l=l: groups(a0, d, drop={f"{l}.attn", f"{l}.mlp"})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    a0, a1 = state(A0_DIR), state(A1_DIR)
    d = {n: a1[n] - a0[n] for n in a0}
    items = prep("dev", args.eval_n)
    out = HERE / "results" / f"grafts_dev{args.eval_n}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["config"] for l in out.read_text().splitlines()} if out.exists() else set()
    total = sum(v.numel() for v in d.values())
    for name, make in configs(a0, a1, d).items():
        if (args.only and name not in args.only) or name in done:
            continue
        t = time.time()
        sd, n_par = make()
        changed = sum(int((sd[n] != a0[n]).sum()) for n in a0)
        row = {"config": name, "params_grafted": int(n_par), "entries_changed": changed,
               "fraction_of_all": changed / total, **metrics(items, score(build(sd), items)),
               "seconds": round(time.time() - t)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
