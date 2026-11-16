"""exp11 readings (PROTOCOL.md) from results/grafts_dev300.jsonl and results/grafts_test.jsonl -> results/results.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
B = 5000


def load(split):
    return {
        j["config"]: j for j in map(json.loads, (HERE / "results" / f"grafts_{split}.jsonl").read_text().splitlines())
    }


def rec(g, b, f):
    den = f.mean() - b.mean()
    return float((g.mean() - b.mean()) / den) if den > 0 else 0.0


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return {"only_x": b, "only_y": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def main():
    dev, test = load("dev300"), load("test")
    rng, rep = np.random.default_rng(0), {}
    for f in ("it", "dom"):
        H = {c[len(f) + 1 :]: np.array(j["hit5"]) for c, j in dev.items() if c.startswith(f + "_")}
        b, u = H["base"], H["full"]
        gain = u.mean() - b.mean()
        r = {k: round(rec(v, b, u), 4) for k, v in H.items() if k not in ("base", "full")}
        rs = next((k for k in (1, 2, 4, 8, 16) if r[f"r{k}"] >= 0.8), None) if gain >= 0.05 else None
        layers = sorted((f"L{i}" for i in range(6)), key=lambda L: -r[L])
        out = {
            "dev_base": round(float(b.mean()), 4),
            "dev_full": round(float(u.mean()), 4),
            "dev_gain": round(float(gain), 4),
            "dev_recovery": r,
            "rstar": rs,
            "layer_rank": layers,
            "same_layers_as_hotpotqa": set(layers[:2]) == {"L3", "L4"},
            "control_passes": r["rand_r4"] < 0.5 * r["r4"],
        }
        if rs and all(f"{f}_{c}" in test for c in ("base", "full", f"r{rs}")):
            T = {c: np.array(test[f"{f}_{c}"]["hit5"]) for c in ("base", "full", f"r{rs}")}
            assert test[f"{f}_base"]["ids"] == test[f"{f}_full"]["ids"] == test[f"{f}_r{rs}"]["ids"]
            g, b0, f0 = T[f"r{rs}"], T["base"], T["full"]
            n = len(g)
            bs = [rec(g[i], b0[i], f0[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
            ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            out["test"] = {
                "base": round(float(b0.mean()), 4),
                "full": round(float(f0.mean()), 4),
                f"r{rs}": round(float(g.mean()), 4),
                "recovery": round(rec(g, b0, f0), 3),
                "recovery_ci95": ci,
                "vs_base": mcnemar(g, b0),
                "vs_full": mcnemar(g, f0),
                "recall_at_k": {c: test[f"{f}_{c}"]["recall"] for c in ("base", "full", f"r{rs}")},
            }
            if f == "it":
                out["primary_holds"] = bool(rs <= 8 and ci[0] > 0.5)
        rep[f] = out
    (HERE / "results" / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
