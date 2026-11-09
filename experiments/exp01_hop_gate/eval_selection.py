"""Offline selection evaluation (no LLM): gold-paragraph recall vs prompt tokens.

1. Tune on DEV (1000 validation questions disjoint from the paper's 400).
2. Pick operating points on DEV only.
3. Report them once on TEST (the paper's 400 questions).

Usage:  uv run python eval_selection.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from common import (
    RESULTS,
    build_retrieval_prompt,
    embed_rows,
    gold_titles,
    jaccard_matrix,
    load_split,
    n_tokens,
    paragraphs_of,
    select_hop_gate,
    select_noise_gate,
    select_topk,
)


def prepare(rows, tag):
    embs = embed_rows(rows, tag)
    out = []
    for r, e in zip(rows, embs, strict=True):
        titles, paras = paragraphs_of(r)
        gold = gold_titles(r)
        base = n_tokens(build_retrieval_prompt(r["question"], []))
        ptok = [n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)]
        e = dict(e)
        e["jac"] = jaccard_matrix(paras)
        out.append(
            {
                "id": r["id"],
                "q": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "titles": titles,
                "paras": paras,
                "gold_idx": [i for i, t in enumerate(titles) if t in gold],
                "e": e,
                "base": base,
                "ptok": ptok,
            }
        )
    return out


def score(items, fn):
    full, rec, prec, npar, tok = [], [], [], [], []
    for it in items:
        sel = fn(it)
        g = set(it["gold_idx"])
        hit = len(g & set(sel))
        full.append(float(hit == len(g)))
        rec.append(hit / len(g))
        prec.append(hit / len(sel) if sel else 0.0)
        npar.append(len(sel))
        tok.append(it["base"] + sum(it["ptok"][i] for i in sel))
    return {
        "full_recall": np.mean(full),  # both gold paragraphs kept
        "gold_recall": np.mean(rec),
        "precision": np.mean(prec),
        "n_paras": np.mean(npar),
        "tokens": np.mean(tok),
    }


def configs():
    yield ("all_k10", {}, lambda it: list(range(len(it["paras"]))))
    yield ("oracle_gold", {}, lambda it: list(it["gold_idx"]))
    for k in range(1, 10):
        yield ("topk", {"k": k}, lambda it, k=k: select_topk(it["e"], k))
    for tau in np.round(np.arange(0.10, 0.71, 0.025), 3):
        yield (
            "noise_gate",
            {"tau": float(tau)},
            lambda it, t=tau: select_noise_gate(it["e"], it["paras"], t),
        )
    grid = itertools.chain(
        itertools.product(
            ["sum", "concat", "pp"],
            [1, 2],
            np.round(np.arange(0.30, 0.71, 0.05), 3),
            np.round(np.arange(0.30, 0.86, 0.05), 3),
        ),
        # high-recall region (added after the first sweep stopped at 0.30)
        itertools.product(
            ["sum", "concat"],
            [1, 2, 3],
            np.round(np.arange(0.15, 0.46, 0.05), 3),
            np.round(np.arange(0.10, 0.30, 0.025), 3),
        ),
    )
    for var, na, t1, t2 in grid:
        yield (
            f"hop_{var}",
            {"tau1": float(t1), "tau2": float(t2), "n_anchor": na},
            lambda it, v=var, a=na, t1=t1, t2=t2: select_hop_gate(
                it["e"], it["paras"], t1, t2, variant=v, n_anchor=a
            ),
        )


def run(items):
    rows = []
    for name, params, fn in configs():
        rows.append({"method": name, "params": json.dumps(params), **score(items, fn)})
    return pd.DataFrame(rows)


def pareto(df):
    df = df.sort_values(["tokens", "full_recall"], ascending=[True, False])
    best, keep = -1.0, []
    for _, r in df.iterrows():
        if r.full_recall > best + 1e-9:
            keep.append(r)
            best = r.full_recall
    return pd.DataFrame(keep)


def pick_points(dev):
    """Operating points chosen on DEV only."""
    ng = dev[dev.method == "noise_gate"]
    hop = dev[dev.method.str.startswith("hop_")]
    picks = []
    for target in (0.85, 0.90, 0.95):  # full-recall targets
        for fam, df in (("noise_gate", ng), ("hop", hop)):
            ok = df[df.full_recall >= target]
            if len(ok):
                r = ok.sort_values("tokens").iloc[0]
                picks.append({"target_full_recall": target, "family": fam, **r.to_dict()})
    return pd.DataFrame(picks)


def fn_from(method, params):
    p = json.loads(params)
    if method == "noise_gate":
        return lambda it: select_noise_gate(it["e"], it["paras"], p["tau"])
    if method.startswith("hop_"):
        v = method.split("_", 1)[1]
        return lambda it: select_hop_gate(
            it["e"], it["paras"], p["tau1"], p["tau2"], variant=v, n_anchor=p["n_anchor"]
        )
    raise ValueError(method)


def main():
    test_rows, dev_rows = load_split()
    print(f"test={len(test_rows)} dev={len(dev_rows)}")
    dev = prepare(dev_rows, "dev")
    test = prepare(test_rows, "test")

    dev_df = run(dev)
    dev_df.to_csv(RESULTS / "dev_grid.csv", index=False)

    picks = pick_points(dev_df)
    out = []
    for _, r in picks.iterrows():
        s = score(test, fn_from(r.method, r.params))
        out.append(
            {
                "target_full_recall": r.target_full_recall,
                "method": r.method,
                "params": r.params,
                "dev_full_recall": r.full_recall,
                "dev_tokens": r.tokens,
                **{f"test_{k}": v for k, v in s.items()},
            }
        )
    for name, fn in (
        ("all_k10", lambda it: list(range(len(it["paras"])))),
        ("oracle_gold", lambda it: list(it["gold_idx"])),
        ("noise_gate_tau0.30", lambda it: select_noise_gate(it["e"], it["paras"], 0.30)),
    ):
        out.append({"method": name, **{f"test_{k}": v for k, v in score(test, fn).items()}})
    test_df = pd.DataFrame(out)
    test_df.to_csv(RESULTS / "test_operating_points.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\nDEV pareto (noise_gate):")
    print(pareto(dev_df[dev_df.method == "noise_gate"]).round(3).to_string(index=False))
    print("\nDEV pareto (all hop variants):")
    print(pareto(dev_df[dev_df.method.str.startswith("hop_")]).round(3).to_string(index=False))
    print("\nTEST operating points (chosen on dev):")
    print(test_df.round(3).to_string(index=False))

    # save per-question selections for the LLM pilot
    sel = {}
    for _, r in picks.iterrows():
        key = f"{r.family}@{r.target_full_recall}"
        fn = fn_from(r.method, r.params)
        sel[key] = {"method": r.method, "params": r.params, "sel": {it["id"]: fn(it) for it in test}}
    sel["noise_gate_tau0.30"] = {
        "method": "noise_gate",
        "params": json.dumps({"tau": 0.30}),
        "sel": {it["id"]: select_noise_gate(it["e"], it["paras"], 0.30) for it in test},
    }
    with open(RESULTS / "test_selections.json", "w") as f:
        json.dump(sel, f)


if __name__ == "__main__":
    main()
