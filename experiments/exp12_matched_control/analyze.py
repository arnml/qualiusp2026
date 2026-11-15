"""exp12 pre-registered analysis (PROTOCOL.md). Written before any exp12 model was trained.

  python analyze.py --rstar          # print r* from dev300 (used by the notebook), or 'none'
  python analyze.py                  # all tests -> results/results.json
RESULTS_DIR (default results/) must hold eval_test.jsonl and eval_dev300.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from data import gold_mask, hops, load

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
E9 = HERE.parent / "exp09_iterative_rerank" / "results"
KS = (2, 3, 4, 5, 6, 8, 10, 15)
B = 5000


def hits(split, orders_by_id, k=5):
    rows = load(split)
    assert set(orders_by_id) == {r["id"] for r in rows}, "ids differ from split"
    return np.array([int(np.array(gold_mask(r), bool)[orders_by_id[r["id"]][:k]].sum() == sum(gold_mask(r))) for r in rows])


def evals(split):
    out = {}
    for line in (RES / f"eval_{split}.jsonl").read_text().splitlines():
        j = json.loads(line)
        out[j["config"]] = dict(zip(j["ids"], j["orders"]))
    return out


def exp09(arm, split):
    return {j["id"]: j["order"] for j in map(json.loads, (E9 / f"rank_{arm}_{split}.jsonl").read_text().splitlines())}


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"only_x": b, "only_y": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def boot_diff(x, y, rng):
    n = len(x); bs = [(x[i] - y[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def recovery(g, base, full):
    den = full.mean() - base.mean()
    return float((g.mean() - base.mean()) / den) if den > 0 else 0.0  # PROTOCOL: denominator <= 0 counts as 0


def rstar(dev):
    base, full = hits("dev300", dev["ctrl_iter"]), hits("dev300", dev["iter_iter"])
    gain = full.mean() - base.mean()
    rec = {r: recovery(hits("dev300", dev[f"g_r{r}"]), base, full) for r in (1, 2, 4, 8) if f"g_r{r}" in dev}
    rs = next((r for r in (1, 2, 4, 8) if rec.get(r, -1) >= 0.8), None) if gain >= 0.05 else None
    return rs, gain, rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rstar", action="store_true"); args = ap.parse_args()
    dev = evals("dev300")
    rs, gain, rec = rstar(dev)
    if args.rstar:
        print(rs if rs else "none"); return
    test, rng = evals("test"), np.random.default_rng(0)
    H = {c: hits("test", o) for c, o in test.items()}
    rep = {}
    x, y = H["iter_iter"], H["ctrl_oneshot"]
    rep["primary"] = {"recall5_iter_iter": round(float(x.mean()), 4), "recall5_ctrl_oneshot": round(float(y.mean()), 4),
                      "diff": round(float((x - y).mean()), 4), "diff_ci95": boot_diff(x, y, rng), **mcnemar(x, y)}
    rep["primary"]["success"] = bool(x.mean() > y.mean() and rep["primary"]["p"] < 0.05)
    seeds = {0: round(float((x - y).mean()), 4)}
    for s in (1, 2):
        if f"iter_iter_s{s}" in H and f"ctrl_oneshot_s{s}" in H:
            seeds[s] = round(float((H[f"iter_iter_s{s}"] - H[f"ctrl_oneshot_s{s}"]).mean()), 4)
    rep["replication"] = {"diff_by_seed": seeds, "mean": round(float(np.mean(list(seeds.values()))), 4),
                          "sd": round(float(np.std(list(seeds.values()), ddof=1)), 4) if len(seeds) > 1 else None,
                          "replicates": bool(len(seeds) == 3 and all(v > 0 for v in seeds.values()))}
    rep["two_by_two_recall5"] = {c: round(float(H[c].mean()), 4) for c in ("iter_iter", "iter_oneshot", "ctrl_iter", "ctrl_oneshot") if c in H}
    m1 = hits("test", exp09("M1", "test")); m1i = hits("test", exp09("M1-iter", "test"))
    rep["descriptive"] = {"exp09_cpu_M1-iter_vs_ctrl_oneshot": {"diff": round(float((m1i - y).mean()), 4), **mcnemar(m1i, y)},
                          "ctrl_oneshot_vs_exp09_M1 (more pairs)": {"diff": round(float((y - m1).mean()), 4), **mcnemar(y, m1)},
                          "gpu_vs_cpu_M1-iter": {"diff": round(float((x - m1i).mean()), 4), **mcnemar(x, m1i)}}
    rows = load("test"); hp = np.array([hops(r) for r in rows])
    rep["recall_at_k"] = {c: {k: round(float(hits("test", test[c], k).mean()), 4) for k in KS} for c in test}
    rep["by_hops_recall5"] = {c: {int(h): round(float(H[c][hp == h].mean()), 3) for h in (2, 3, 4)} for c in H}
    sec = {"dev_gain": round(float(gain), 4), "dev_recovery": {r: round(v, 3) for r, v in rec.items()}, "rstar": rs}
    if gain < 0.05:
        sec["status"] = "not testable (dev gain < 0.05)"
    else:
        r2, rr = hits("dev300", dev["g_r2"]), hits("dev300", dev["g_rand_r2"])
        bd, fd = hits("dev300", dev["ctrl_iter"]), hits("dev300", dev["iter_iter"])
        sec["control"] = {"rand_r2": round(recovery(rr, bd, fd), 3), "r2": round(recovery(r2, bd, fd), 3)}
        sec["control"]["passes"] = bool(recovery(rr, bd, fd) < 0.5 * recovery(r2, bd, fd))
        if rs is None:
            sec["status"] = "fails (no rank <= 8 reaches 80% on dev300)"
        else:
            g, b0, f0 = H[f"g_r{rs}"], H["ctrl_iter"], H["iter_iter"]; n = len(g)
            bs = [recovery(g[i], b0[i], f0[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
            ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            sec.update({"test_recovery": round(recovery(g, b0, f0), 3), "test_recovery_ci95": ci,
                        "test_gain": round(float(f0.mean() - b0.mean()), 4), "test_gain_ci95": boot_diff(f0, b0, rng)})
            sec["status"] = "holds" if ci[0] > 0.5 and sec["control"]["passes"] else "fails (CI lower bound <= 0.5 or control)"
    rep["secondary"] = sec
    (RES / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
