"""exp14 pre-registered analysis (PROTOCOL.md). Written before any exp14 ranking was run.

python analyze.py      # RESULTS_DIR: rank_{b_train,b_tune,b_eval,router_train}.jsonl (+ optional rank_dev_fresh.jsonl)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from data import gold_mask, hops, load
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
B, KS, MAX_PASSES = 5000, (2, 3, 4, 5, 10, 15), 50.0
WH = ("who", "what", "which", "where", "when", "how")
SHA = {
    "iter_iter": "2c91c40f1310ca02",
    "ctrl_oneshot": "ed87244f462412f9",
    "iter_oneshot": "2c91c40f1310ca02",
    "ctrl_iter": "ed87244f462412f9",
}
# Deviation 1: the notebook writes the hashes of the models actually used (re-trained) to model_sha.json
if (RES / "model_sha.json").exists():
    _m = json.loads((RES / "model_sha.json").read_text())
    SHA = {
        "iter_iter": _m["m1_iter_s0"][:16],
        "ctrl_oneshot": _m["m1_ctrl"][:16],
        "iter_oneshot": _m["m1_iter_s0"][:16],
        "ctrl_iter": _m["m1_ctrl"][:16],
    }
CONFIGS = tuple(SHA)


def ranks(split):
    rows = load(split)
    out = {}
    for line in (RES / f"rank_{split}.jsonl").read_text().splitlines():
        j = json.loads(line)
        if j["config"] not in CONFIGS:
            continue
        assert j["ids"] == [r["id"] for r in rows], f"ids differ: {split}"
        assert j["sha256"][:16] == SHA[j["config"]], f"model hash mismatch: {split} {j['config']}"
        assert len(j["orders"]) == len(j["passes"]) == len(j["top10"]) == len(rows), f"length mismatch: {split}"
        assert j["config"] not in out, f"duplicate {split} {j['config']}"
        assert all(sorted(o) == list(range(len(r["paragraphs"]))) for o, r in zip(j["orders"], rows))
        out[j["config"]] = j
    return rows, out


def hits(rows, orders, k=5):
    return np.array([int(np.array(gold_mask(r), bool)[o[:k]].sum() == sum(gold_mask(r))) for r, o in zip(rows, orders)])


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"only_x": b, "only_y": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def boot(d, rng):
    n = len(d)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def feats_benefit(rows, one):
    """PROTOCOL: question text features + the one-shot model's score profile (available after the one-shot pass)."""
    F = []
    for r, t in zip(rows, one["top10"]):
        q = r["question"]
        w = q.lower().split()
        s = np.array(t, float)
        p = np.exp(s - s.max())
        p /= p.sum()
        F.append(
            [
                len(w),
                q.count(","),
                q.lower().count(" of "),
                q.count("'s"),
                sum(x in WH for x in w),
                sum(x[:1].isupper() for x in q.split()[1:]),
                *s,
                *(s[:5] - s[1:6]),
                float(-(p * np.log(p)).sum()),
            ]
        )
    return np.array(F, float)


def feats_text(rows):  # descriptive: text-only benefit router
    return np.array(
        [
            [
                len(r["question"].split()),
                r["question"].count(","),
                r["question"].lower().count(" of "),
                r["question"].count("'s"),
                sum(x in WH for x in r["question"].lower().split()),
                sum(x[:1].isupper() for x in r["question"].split()[1:]),
            ]
            for r in rows
        ],
        float,
    )


def feats_hop(rows, one):  # exp13's router features (comparison)
    return np.array(
        [[len(r["question"].split()), r["question"].count(",")] + t for r, t in zip(rows, one["top10"])], float
    )


def cascade(rows, R, route):
    one, itr = R["ctrl_oneshot"], R["iter_iter"]
    orders = [itr["orders"][i] if route[i] else one["orders"][i] for i in range(len(rows))]
    passes = np.array(one["passes"]) + np.where(route, np.array(itr["passes"]), 0)
    return orders, float(passes.mean())


def main():
    rng, rep = (
        np.random.default_rng(0),
        {
            "model_sha16": {"m1_iter_s0": SHA["iter_iter"], "m1_ctrl": SHA["ctrl_oneshot"]},
            "models_registered": SHA["iter_iter"] == "2c91c40f1310ca02" and SHA["ctrl_oneshot"] == "ed87244f462412f9",
        },
    )
    D = {sp: ranks(sp) for sp in ("b_train", "b_tune", "b_eval", "router_train")}
    gain = {
        sp: hits(rows, R["iter_iter"]["orders"]) - hits(rows, R["ctrl_oneshot"]["orders"])
        for sp, (rows, R) in D.items()
        if sp != "router_train"
    }

    # Benefit router: gradient-boosted regression of g = iter_hit5 - oneshot_hit5, fit on b_train
    Xtr = feats_benefit(*(D["b_train"][0], D["b_train"][1]["ctrl_oneshot"]))
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(Xtr, gain["b_train"])
    pred = {sp: reg.predict(feats_benefit(D[sp][0], D[sp][1]["ctrl_oneshot"])) for sp in ("b_tune", "b_eval")}

    # tau on b_tune: max cascade recall@5 with mean passes <= 50 (ties -> fewer routed)
    rows, R = D["b_tune"]
    best = None
    for tau in np.unique(np.quantile(pred["b_tune"], np.linspace(0, 1, 101))):
        route = pred["b_tune"] >= tau
        orders, ps = cascade(rows, R, route)
        if ps <= MAX_PASSES:
            key = (hits(rows, orders).mean(), -route.mean())
            if best is None or key > best[0]:
                best = (key, float(tau))
    tau = best[1] if best else float("inf")
    rep["router"] = {"tau": tau, "b_tune_recall5": round(best[0][0], 4) if best else None}
    # descriptive: tau stability under bootstrap of b_tune
    taus, fr = [], []
    for _ in range(200):
        ix = rng.integers(0, len(rows), len(rows))
        bb = None
        for t in np.unique(np.quantile(pred["b_tune"][ix], np.linspace(0, 1, 101))):
            route = pred["b_tune"][ix] >= t
            orders = [
                R["iter_iter"]["orders"][j] if route[k] else R["ctrl_oneshot"]["orders"][j] for k, j in enumerate(ix)
            ]
            ps = (
                np.array(R["ctrl_oneshot"]["passes"])[ix] + np.where(route, np.array(R["iter_iter"]["passes"])[ix], 0)
            ).mean()
            if ps <= MAX_PASSES:
                key = (hits([rows[j] for j in ix], orders).mean(), -route.mean())
                if bb is None or key > bb[0]:
                    bb = (key, float(t), float(route.mean()))
        if bb:
            taus.append(bb[1])
            fr.append(bb[2])
    rep["router"]["tau_boot_q05_q50_q95"] = [round(float(np.quantile(taus, q)), 4) for q in (0.05, 0.5, 0.95)]
    rep["router"]["routed_frac_boot_q05_q50_q95"] = [round(float(np.quantile(fr, q)), 3) for q in (0.05, 0.5, 0.95)]
    reg_text = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(
        feats_text(D["b_train"][0]), gain["b_train"]
    )

    # exp13-style hop router (comparison), refit exactly as exp13 on router_train
    rt_rows, rt = D["router_train"]
    Fh = feats_hop(rt_rows, rt["ctrl_oneshot"])
    yh = np.array([int(hops(r) == 4) for r in rt_rows])
    sc = StandardScaler().fit(Fh)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Fh), yh)

    # Evaluation on b_eval
    rows, R = D["b_eval"]
    hp = np.array([hops(r) for r in rows])
    o, i = hits(rows, R["ctrl_oneshot"]["orders"]), hits(rows, R["iter_iter"]["orders"])
    routes = {
        "benefit": pred["b_eval"] >= tau,
        "hop_exp13": lr.predict_proba(sc.transform(feats_hop(rows, R["ctrl_oneshot"])))[:, 1] >= 0.5,
        "oracle_benefit": gain["b_eval"] > 0,
    }
    rep["b_eval"] = {
        "n": len(rows),
        "oneshot_recall5": round(float(o.mean()), 4),
        "iterative_recall5": round(float(i.mean()), 4),
        "iterative_gain": round(float((i - o).mean()), 4),
        "iterative_gain_ci95": boot(i - o, rng),
        "benefit_auc": round(float(roc_auc_score(gain["b_eval"] > 0, pred["b_eval"])), 4)
        if (gain["b_eval"] > 0).any()
        else None,
    }
    H = {}
    for name, route in routes.items():
        orders, ps = cascade(rows, R, route)
        H[name] = hits(rows, orders)
        rep["b_eval"][name] = {
            "recall5": round(float(H[name].mean()), 4),
            "passes": round(ps, 1),
            "routed": round(float(route.mean()), 3),
            "recall_at_k": {k: round(float(hits(rows, orders, k).mean()), 4) for k in KS},
            "by_hops_recall5": {int(h): round(float(H[name][hp == h].mean()), 3) for h in (2, 3, 4)},
        }
    c = H["benefit"]
    gate = float((i - o).mean()) >= 0.02
    pr = {
        "gate_iterative_gain_ge_0.02": gate,
        "diff": round(float((c - o).mean()), 4),
        "diff_ci95": boot(c - o, rng),
        **mcnemar(c, o),
        "passes": rep["b_eval"]["benefit"]["passes"],
    }
    pr["success"] = bool(
        gate and pr["diff"] >= 0.01 and pr["p"] < 0.05 and pr["passes"] <= MAX_PASSES
    )  # PROTOCOL (after review)
    pr["status"] = "not testable (iterative gain < 0.02)" if not gate else ("holds" if pr["success"] else "fails")
    rep["primary"] = pr
    h = H["hop_exp13"]
    rep["secondary"] = {
        "benefit_vs_hop_router": {
            "diff": round(float((c - h).mean()), 4),
            "diff_ci95": boot(c - h, rng),
            **mcnemar(c, h),
        },
        "benefit_vs_iterative": {"diff": round(float((c - i).mean()), 4), "diff_ci95": boot(c - i, rng)},
    }
    s = rep["secondary"]
    s["beats_hop_router"] = bool(c.mean() > h.mean() and s["benefit_vs_hop_router"]["p"] < 0.05)
    d = c - i
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    s["benefit_vs_iterative"]["one_sided_95_lower"] = float(np.percentile(bs, 5))
    s["noninferior_to_iterative"] = bool(s["benefit_vs_iterative"]["one_sided_95_lower"] > -0.01)
    s["label"] = "exploratory (secondary; no multiplicity correction)"
    # descriptive: 2x2, truncation split, text-only router AUC
    rep["b_eval"]["two_by_two_recall5"] = {
        k: round(float(hits(rows, R[k]["orders"]).mean()), 4) for k in CONFIGS if k in R
    }
    long_ = np.array(R["iter_iter"]["max_query_tokens"]) >= 512
    rep["b_eval"]["primary_diff_by_truncation"] = {
        "iter_query_ge_512": {
            "n": int(long_.sum()),
            "diff": round(float((c - o)[long_].mean()), 4) if long_.any() else None,
        },
        "below_512": {"n": int((~long_).sum()), "diff": round(float((c - o)[~long_].mean()), 4)},
    }
    pt = reg_text.predict(feats_text(rows))
    rep["b_eval"]["text_only_auc"] = (
        round(float(roc_auc_score(gain["b_eval"] > 0, pt)), 4) if (gain["b_eval"] > 0).any() else None
    )

    # Descriptive: the same frozen router + tau on dev_fresh (re-used set from exp13; labelled re-use)
    if (RES / "rank_dev_fresh.jsonl").exists():
        rows, R = ranks("dev_fresh")
        route = reg.predict(feats_benefit(rows, R["ctrl_oneshot"])) >= tau
        orders, ps = cascade(rows, R, route)
        o2, i2, c2 = (hits(rows, R["ctrl_oneshot"]["orders"]), hits(rows, R["iter_iter"]["orders"]), hits(rows, orders))
        rep["dev_fresh_reuse"] = {
            "oneshot": round(float(o2.mean()), 4),
            "iterative": round(float(i2.mean()), 4),
            "benefit_cascade": round(float(c2.mean()), 4),
            "passes": round(ps, 1),
            "routed": round(float(route.mean()), 3),
            "cascade_vs_oneshot": mcnemar(c2, o2),
        }
    (RES / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
