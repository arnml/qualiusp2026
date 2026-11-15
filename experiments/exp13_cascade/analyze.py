"""exp13 pre-registered analysis (PROTOCOL.md). Written before any exp13 ranking was run.

  python analyze.py        # RESULTS_DIR must hold rank_{f4_dev,f4_train,router_train,dev_fresh}.jsonl -> results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import gold_mask, hops, load

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
B, KS = 5000, (2, 3, 4, 5, 6, 8, 10, 15)


def ranks(split):
    rows = load(split)
    out = {}
    for line in (RES / f"rank_{split}.jsonl").read_text().splitlines():
        j = json.loads(line)
        assert j["ids"] == [r["id"] for r in rows], f"ids differ: {split} {j['config']}"
        assert j["config"] not in out, f"duplicate config {split} {j['config']}"
        assert all(sorted(o) == list(range(len(r["paragraphs"]))) for o, r in zip(j["orders"], rows)), "bad order"
        out[j["config"]] = j
    return rows, out


def hits(rows, orders, k=5):
    return np.array([int(np.array(gold_mask(r), bool)[o[:k]].sum() == sum(gold_mask(r))) for r, o in zip(rows, orders)])


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"only_x": b, "only_y": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def boot(d, rng):
    n = len(d); bs = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]  # unrounded (PROTOCOL: decisions use these)


def feats(rows, oneshot):
    """Router features (PROTOCOL): question length in words, commas, and the one-shot model's sorted top-10 scores."""
    return np.array([[len(r["question"].split()), r["question"].count(",")] + t for r, t in zip(rows, oneshot["top10"])], float)


def main():
    rng, rep = np.random.default_rng(0), {}
    # 1. Primary: 4-hop, iterative M1-iter vs one-shot M1-ctrl (seed 0), pooled over f4_dev + f4_train
    X, Y, strata, seeds = [], [], {}, {s: ([], []) for s in ("", "_s1", "_s2")}
    for sp in ("f4_dev", "f4_train"):
        rows, R = ranks(sp)
        assert all(hops(r) == 4 for r in rows)
        x, y = hits(rows, R["iter_iter"]["orders"]), hits(rows, R["ctrl_oneshot"]["orders"])
        X.append(x); Y.append(y)
        strata[sp] = {"n": len(x), "iter": round(float(x.mean()), 4), "ctrl": round(float(y.mean()), 4),
                      "diff": round(float((x - y).mean()), 4), **mcnemar(x, y)}
        for s in seeds:
            if f"iter_iter{s}" in R and f"ctrl_oneshot{s}" in R:
                seeds[s][0].append(hits(rows, R[f"iter_iter{s}"]["orders"])); seeds[s][1].append(hits(rows, R[f"ctrl_oneshot{s}"]["orders"]))
    x, y = np.concatenate(X), np.concatenate(Y)
    p = {"n": len(x), "iter": round(float(x.mean()), 4), "ctrl": round(float(y.mean()), 4),
         "diff": round(float((x - y).mean()), 4), "diff_ci95": boot(x - y, rng), **mcnemar(x, y), "strata": strata}
    p["both_strata_positive"] = all(v["diff"] > 0 for v in strata.values())
    p["success"] = bool(x.mean() > y.mean() and p["p"] < 0.05 and p["both_strata_positive"])  # PROTOCOL (after review)
    two = {}
    for sp in ("f4_dev", "f4_train"):
        rows, R = ranks(sp)
        two[sp] = {c: round(float(hits(rows, R[c]["orders"]).mean()), 4) for c in ("iter_iter", "iter_oneshot", "ctrl_iter", "ctrl_oneshot") if c in R}
        two[sp]["iter_iter_max_query_tokens_ge_512"] = round(float(np.mean(np.array(R["iter_iter"]["max_query_tokens"]) >= 512)), 3)
    p["two_by_two_recall5"] = two
    p["seed_diffs"] = {s or "_s0": round(float((np.concatenate(a) - np.concatenate(b)).mean()), 4) for s, (a, b) in seeds.items() if a}
    rep["primary_4hop"] = p

    # 2. Router: logistic regression for "4-hop", fit on router_train (one-shot features), threshold 0.5
    rt_rows, rt = ranks("router_train")
    Ft, yt = feats(rt_rows, rt["ctrl_oneshot"]), np.array([int(hops(r) == 4) for r in rt_rows])
    sc = StandardScaler().fit(Ft); lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Ft), yt)  # 4-hop is 6.3% of router_train

    # 3. Cascade on dev_fresh (natural hop mix)
    rows, R = ranks("dev_fresh")
    hp = np.array([hops(r) for r in rows]); oneshot, itr = R["ctrl_oneshot"], R["iter_iter"]
    route = lr.predict_proba(sc.transform(feats(rows, oneshot)))[:, 1] >= 0.5
    pol = {"oneshot": np.zeros(len(rows), bool), "iterative": np.ones(len(rows), bool), "cascade": route, "oracle_cascade": hp == 4}
    po, pi = np.array(oneshot["passes"]), np.array(itr["passes"])
    H = {}
    for name, m in pol.items():
        orders = [itr["orders"][i] if m[i] else oneshot["orders"][i] for i in range(len(rows))]
        passes = np.where(m, (po + pi) if name in ("cascade", "oracle_cascade") else pi, po)  # a routed question pays one-shot + iterative
        H[name] = hits(rows, orders)
        rep.setdefault("cascade", {})[name] = {
            "recall_at_k": {k: round(float(hits(rows, orders, k).mean()), 4) for k in KS},
            "passes": round(float(passes.mean()), 1), "routed": round(float(m.mean()), 3),
            "by_hops_recall5": {int(h): round(float(H[name][hp == h].mean()), 3) for h in (2, 3, 4)}}
    rep["router"] = {"train_4hop_rate": round(float(yt.mean()), 3), "dev_fresh_routed_4hop_recall": round(float(route[hp == 4].mean()), 3),
                     "dev_fresh_routed_precision": round(float((hp[route] == 4).mean()), 3) if route.any() else None}
    # Secondary (PROTOCOL): cascade vs one-shot, recall@5 McNemar; cascade vs iterative, non-inferiority margin 0.01
    c, o, i = H["cascade"], H["oneshot"], H["iterative"]
    rest = ~np.isin([r["id"] for r in rows], [r["id"] for r in load("f4_dev")])
    rep["cascade_sensitivity_without_f4_dev"] = {n: round(float(H[n][rest].mean()), 4) for n in H} | {"n": int(rest.sum())}
    rep["secondary"] = {"cascade_vs_oneshot": {"diff": round(float((c - o).mean()), 4), "diff_ci95": boot(c - o, rng), **mcnemar(c, o)},
                        "cascade_vs_iterative": {"diff": round(float((c - i).mean()), 4), "diff_ci95": boot(c - i, rng)}}
    s = rep["secondary"]
    s["cascade_beats_oneshot"] = bool(c.mean() > o.mean() and s["cascade_vs_oneshot"]["p"] < 0.05)
    s["cascade_noninferior_to_iterative"] = bool(s["cascade_vs_iterative"]["diff_ci95"][0] > -0.01)
    s["cascade_useful"] = bool(s["cascade_noninferior_to_iterative"] and rep["cascade"]["cascade"]["passes"] <= 50)
    (RES / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
