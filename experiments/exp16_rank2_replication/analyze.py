"""exp16 pre-registered analysis (PROTOCOL.md). Written before any exp16 model was trained.

  python analyze.py      # RESULTS_DIR: eval_{hp,tw}_{tune,test}.jsonl -> results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from data import load

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
B = 5000
RANKS = (1, 2, 4, 8)


QL = {}


def evals(split):
    rows = load(split); out = {}
    for line in (RES / f"eval_{split}.jsonl").read_text().splitlines():
        j = json.loads(line)
        assert j["ids"] == [r["id"] for r in rows], f"ids differ {split}"
        assert j["config"] not in out, f"duplicate {split} {j['config']}"
        out[j["config"]] = j["orders"]
        if j["config"] == "iter_iter":
            QL[split] = j.get("max_query_tokens", [])
    return rows, out


def hits(rows, orders, slack=1, k=None):
    """Full-support recall: all gold paragraphs within the top (|gold| + slack), or within the top k if k is given."""
    res = []
    for r, o in zip(rows, orders):
        g = np.array(r["gold"], bool); kk = k if k else int(g.sum()) + slack
        res.append(int(g[o[:kk]].sum() == g.sum()))
    return np.array(res)


def recovery(g, base, full):
    den = full.mean() - base.mean()
    return float((g.mean() - base.mean()) / den) if den > 0 else 0.0


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"only_x": b, "only_y": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def check_exclusion():
    """PROTOCOL: HotpotQA evaluation ids exclude every id used in exp01-exp07."""
    used = set()
    for f in ("exp01_hop_gate", "exp02_cross_encoder", "exp07_chain_selector"):
        p = HERE.parent / f / "split_ids.json"
        if p.exists():
            for v in json.loads(p.read_text()).values():
                used.update(v)
    ids = json.loads((HERE / "split_ids.json").read_text())
    assert not used & set(ids["hp_test"] + ids["hp_tune"]), "HotpotQA evaluation overlaps earlier experiments"
    return len(used)


def main():
    rng, rep = np.random.default_rng(0), {"excluded_hotpot_ids": check_exclusion()}
    for ds in ("tw", "hp"):
        rows_t, tu = evals(f"{ds}_tune"); rows, te = evals(f"{ds}_test")
        H = lambda R, E, c: hits(R, E[c])
        base_t, full_t = H(rows_t, tu, "ctrl_iter"), H(rows_t, tu, "iter_iter")
        gap = float(full_t.mean() - base_t.mean())
        rec_t = {r: round(recovery(H(rows_t, tu, f"g_r{r}"), base_t, full_t), 3) for r in RANKS}
        rand_t = recovery(H(rows_t, tu, "g_rand_r2"), base_t, full_t)
        d = {"tune": {"gap": round(gap, 4), "recovery": rec_t, "rand_r2": round(rand_t, 3),
                      "rstar_80": next((r for r in RANKS if rec_t[r] >= 0.8), None)}}
        d["gate_gap_ge_0.05"] = gap >= 0.05
        noise_t = recovery(H(rows_t, tu, "g_noise_r2"), base_t, full_t)
        d["tune"]["noise_r2"] = round(noise_t, 3)
        d["control_passes"] = bool(rand_t < 0.5 * rec_t[2] and noise_t < 0.5 * rec_t[2])
        b0, f0 = H(rows, te, "ctrl_iter"), H(rows, te, "iter_iter")
        g2 = H(rows, te, "g_r2"); n = len(g2)
        bs = [recovery(g2[i], b0[i], f0[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        gd = f0 - b0; gbs = [gd[rng.integers(0, n, n)].mean() for _ in range(B)]
        gap_ci = [float(np.percentile(gbs, 2.5)), float(np.percentile(gbs, 97.5))]
        d["test"] = {"ctrl_iter": round(float(b0.mean()), 4), "iter_iter": round(float(f0.mean()), 4), "g_r2": round(float(g2.mean()), 4),
                     "recovery_r2": round(recovery(g2, b0, f0), 3), "recovery_r2_ci95": ci, "gap_ci95": gap_ci,
                     "recovery_r2_recall_at_5": round(recovery(hits(rows, te["g_r2"], k=5), hits(rows, te["ctrl_iter"], k=5), hits(rows, te["iter_iter"], k=5)), 3),
                     "iter_query_ge_512": round(float(np.mean([0] * 0 + [1 if q >= 512 else 0 for q in QL[f"{ds}_test"]])), 3) if f"{ds}_test" in QL else None,
                     "recovery_other_ranks": {r: round(recovery(H(rows, te, f"g_r{r}"), b0, f0), 3) for r in RANKS if f"g_r{r}" in te}}
        ok = d["gate_gap_ge_0.05"] and d["control_passes"] and gap_ci[0] > 0
        d["holds"] = bool(ok and d["test"]["recovery_r2"] >= 0.8 and ci[0] > 0.5)
        d["status"] = ("not testable (tune gap < 0.05 or test gap CI not above 0)" if not (d["gate_gap_ge_0.05"] and gap_ci[0] > 0)
                       else "fails (control)" if not d["control_passes"]
                       else "holds" if d["holds"] else "inconclusive" if d["test"]["recovery_r2"] >= 0.8 else "fails")
        # descriptive: 2x2 and the iteration effect (system comparison), by question type
        two = {c: round(float(H(rows, te, c).mean()), 4) for c in ("iter_iter", "iter_oneshot", "ctrl_iter", "ctrl_oneshot")}
        x, y = f0, H(rows, te, "ctrl_oneshot")
        types = sorted({r["type"] for r in rows}); ty = np.array([r["type"] for r in rows])
        d["descriptive"] = {"two_by_two": two, "iter_vs_oneshot": {"diff": round(float((x - y).mean()), 4), **mcnemar(x, y)},
                            "by_type_iter_minus_oneshot": {t: round(float((x - y)[ty == t].mean()), 4) for t in types},
                            "recall_at_5": {c: round(float(hits(rows, te[c], k=5).mean()), 4) for c in ("iter_iter", "ctrl_oneshot", "g_r2")}}
        rep[ds] = d
    held = [k for k in ("tw", "hp") if rep[k]["holds"]]
    rep["replication"] = {"datasets_holding": held, "replicates": len(held) == 2,
                          "note": "only both-hold counts as replication; a single dataset holding is exploratory",
                          "with_musique_exp12": "MuSiQue (exp12): rank-2 recovery 0.868 [0.777, 0.958], holds"}
    (RES / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
