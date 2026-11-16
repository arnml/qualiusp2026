"""exp10 pre-registered analysis (PROTOCOL.md): stop scorers on holdout, tau on dev300, comparisons on test.
Writes results/results.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from data import build_retrieval_prompt, gold_mask, hops, load, n_tokens, paragraphs_of
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
KS = list(range(2, 21))
TARGETS = (0.90, 0.95)
FIXED_K = {0.90: 10, 0.95: 14}  # exp09's dev300-chosen k for M1-iter
ARMS = ("M1", "M1-iter")


def load_arm(arm, split):
    rows = load(split)
    recs = {
        j["id"]: j
        for j in map(
            json.loads, (HERE / "results" / f"rank_{arm}_{split}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    out = []
    for r in rows:
        j = recs[r["id"]]
        _, paras = paragraphs_of(r)
        g = np.array(gold_mask(r), bool)
        o = j["order"]
        n = len(o)
        base = n_tokens(build_retrieval_prompt(r["question"], []))
        ptok = np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)])
        R = np.array([g[o[: min(k, n)]].sum() == g.sum() for k in KS])
        T = np.array([base + ptok[o[: min(k, n)]].sum() for k in KS])
        cum = np.cumsum(j["step_passes"])
        Pss = np.array([cum[min(k, len(cum) - 1)] if arm == "M1-iter" else cum[-1] for k in KS])  # passes to stop at k
        sel = np.array(j["sel"], float)
        out.append({"id": r["id"], "hops": hops(r), "R": R, "T": T, "P": Pss, "sel": sel})
    return out


def feats(x):
    s = x["sel"]
    s10 = np.pad(s[:10], (0, max(0, 10 - len(s))), mode="edge")
    F = []
    for k in KS:
        kk = min(k, len(s))
        nxt = s[kk] if kk < len(s) else s[-1] - 5.0  # nothing left: a clearly low value
        F.append(np.r_[k, s10, s[kk - 1], nxt, s[kk - 1] - nxt, s[:kk].min()])
    return np.array(F)


def X(data):
    return np.vstack([feats(x) for x in data]), np.concatenate([x["R"] for x in data]).astype(int)


def pick(P, tau):
    ok = P >= tau
    return np.where(ok.any(1), ok.argmax(1), len(KS) - 1)


def tune(P, data, target):
    R = np.array([x["R"] for x in data])
    T = np.array([x["T"] for x in data])
    best = None
    for tau in np.unique(np.quantile(P, np.linspace(0, 1, 801))):
        c = pick(P, tau)
        rec = R[np.arange(len(c)), c].mean()
        tok = T[np.arange(len(c)), c].mean()
        if rec >= target and (best is None or tok < best[2]):
            best = (float(tau), float(rec), float(tok))
    return best


def boot(d, rng, n=5000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    D = {(a, s): load_arm(a, s) for a in ARMS for s in ("holdout", "dev300", "test")}
    rng = np.random.default_rng(0)
    rep = {"policies": {}}
    for a in ARMS:
        Xh, yh = X(D[(a, "holdout")])
        sc = StandardScaler().fit(Xh)
        lr = LogisticRegression(max_iter=5000).fit(sc.transform(Xh), yh)
        for s in ("dev300", "test"):
            Xs, _ = X(D[(a, s)])
            D[(a, s, "p")] = lr.predict_proba(sc.transform(Xs))[:, 1].reshape(-1, len(KS))
    test = D[("M1-iter", "test")]
    H = np.array([x["hops"] for x in test])
    for target in TARGETS:
        per, row = {}, {}
        for a in ARMS:
            b = tune(D[(a, "dev300", "p")], D[(a, "dev300")], target)
            c = pick(D[(a, "test", "p")], b[0])
            dt = D[(a, "test")]
            ix = np.arange(len(c))
            R = np.array([x["R"] for x in dt])[ix, c].astype(float)
            T = np.array([x["T"] for x in dt])[ix, c]
            Pp = np.array([x["P"] for x in dt])[ix, c]
            per[f"adaptive_{a}"] = (R, T, Pp)
            row[f"adaptive_{a}"] = {
                "dev": b,
                "test_recall": round(float(R.mean()), 4),
                "test_tokens": round(float(T.mean()), 1),
                "mean_k": round(float(np.array(KS)[c].mean()), 2),
                "passes": round(float(Pp.mean()), 1),
            }
            # equal-recall frontier on test (descriptive)
            row[f"adaptive_{a}"]["frontier_test"] = tune(D[(a, "test", "p")], dt, target)
        kf = FIXED_K[target] - 2
        R = np.array([x["R"][kf] for x in test], float)
        T = np.array([x["T"][kf] for x in test])
        Pp = np.array([x["P"][kf] for x in test])
        per["fixed_M1-iter"] = (R, T, Pp)
        row["fixed_M1-iter"] = {
            "k": FIXED_K[target],
            "test_recall": round(float(R.mean()), 4),
            "test_tokens": round(float(T.mean()), 1),
            "passes": round(float(Pp.mean()), 1),
        }
        for x_, y_, role in (
            ("adaptive_M1-iter", "fixed_M1-iter", "primary"),
            ("adaptive_M1-iter", "adaptive_M1", "secondary"),
        ):
            dt_, dr_ = per[x_][1] - per[y_][1], per[x_][0] - per[y_][0]
            ct, cr = boot(dt_, rng), boot(dr_, rng)
            row[f"{x_}_vs_{y_}"] = {
                "role": role,
                "tokens": round(float(dt_.mean()), 1),
                "tokens_ci95": ct,
                "recall": round(float(dr_.mean()), 4),
                "recall_ci95": cr,
                "success": bool(ct[1] < 0 and cr[1] >= 0),
            }
        row["by_hops"] = {
            int(h): {
                k: {"recall": round(float(v[0][H == h].mean()), 3), "tokens": round(float(v[1][H == h].mean()))}
                for k, v in per.items()
            }
            for h in (2, 3, 4)
        }
        rep["policies"][f"recall_{target:g}"] = row
    pol = rep["policies"]
    rep["primary_success"] = any(pol[f"recall_{t:g}"]["adaptive_M1-iter_vs_fixed_M1-iter"]["success"] for t in TARGETS)
    rep["secondary_success"] = any(pol[f"recall_{t:g}"]["adaptive_M1-iter_vs_adaptive_M1"]["success"] for t in TARGETS)
    (HERE / "results" / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
