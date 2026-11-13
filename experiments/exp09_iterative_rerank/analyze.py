"""exp09 pre-registered analysis (PROTOCOL.md). Reads results/rank_<arm>_<split>.jsonl, writes results/results.json."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

from data import build_retrieval_prompt, gold_mask, hops, load, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
ARMS = ("A1", "A1-iter", "M1", "M1-iter")
KS = range(1, 21)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (a only, b only, p)."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p


def boot(d, rng, n=5000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def tables(split, arm):
    """R[q, k-1] = full recall of top k, T[q, k-1] = prompt tokens of top k; passes per question."""
    rows = load(split)
    recs = {j["id"]: j for j in map(json.loads, (HERE / "results" / f"rank_{arm}_{split}.jsonl")
                                    .read_text(encoding="utf-8").splitlines())}
    R = np.zeros((len(rows), len(KS)), bool); T = np.zeros((len(rows), len(KS))); P = []
    for n, r in enumerate(rows):
        _, paras = paragraphs_of(r); g = np.array(gold_mask(r), bool); o = recs[r["id"]]["order"]
        base = n_tokens(build_retrieval_prompt(r["question"], []))
        ptok = [n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)]  # same per-paragraph count as exp08
        for c, k in enumerate(KS):
            top = o[:k]; R[n, c] = g[top].sum() == g.sum(); T[n, c] = base + sum(ptok[i] for i in top)
        P.append(recs[r["id"]]["passes"])
    return R, T, np.array(P), np.array([hops(r) for r in rows])


def main():
    D = {(a, s): tables(s, a) for a in ARMS for s in ("dev300", "test")
         if (HERE / "results" / f"rank_{a}_{s}.jsonl").exists()}
    arms = [a for a in ARMS if (a, "test") in D]
    rng = np.random.default_rng(0)
    rep = {"arms": arms,
           "recall_at_k_test": {a: {k: round(float(D[(a, "test")][0][:, k - 1].mean()), 3) for k in (2, 3, 4, 5, 6, 8, 10, 15)}
                                for a in arms},
           "passes_per_question_test": {a: float(D[(a, "test")][2].mean()) for a in arms}}
    H = D[(arms[0], "test")][3]
    rep["recall_at_5_by_hops"] = {int(h): {a: round(float(D[(a, "test")][0][H == h, 4].mean()), 3) for a in arms}
                                  for h in (2, 3, 4)}
    tests = {}
    for x, y, tag in (("M1-iter", "M1", "primary"), ("M1-iter", "A1", "secondary_a"), ("A1-iter", "A1", "descriptive"),
                      ("M1", "A1", "descriptive")):
        if x in arms and y in arms:
            a, b = D[(x, "test")][0][:, 4], D[(y, "test")][0][:, 4]
            n10, n01, p = mcnemar(a, b)
            tests[f"{x}_vs_{y}_recall@5"] = {"role": tag, "x": round(float(a.mean()), 4), "y": round(float(b.mean()), 4),
                                             "x_only": n10, "y_only": n01, "p": p,
                                             "success": bool(a.mean() > b.mean() and p < 0.05)}
    rep["recall@5_tests"] = tests
    # secondary b: k chosen on dev300 per arm, applied to test
    tok = {}
    for target in (0.90, 0.95):
        per = {}
        for a in arms:
            if (a, "dev300") not in D:
                continue
            dev_r = D[(a, "dev300")][0].mean(0); k = int(np.argmax(dev_r >= target)) + 1
            R, T = D[(a, "test")][:2]
            per[a] = (k, R[:, k - 1].astype(float), T[:, k - 1])
        row = {a: {"k": v[0], "test_recall": round(float(v[1].mean()), 4), "test_tokens": round(float(v[2].mean()), 1)}
               for a, v in per.items()}
        for x, y in (("M1-iter", "M1"), ("M1-iter", "A1"), ("M1", "A1")):
            if x in per and y in per:
                dt, dr = per[x][2] - per[y][2], per[x][1] - per[y][1]
                ct, cr = boot(dt, rng), boot(dr, rng)
                row[f"{x}_minus_{y}"] = {"tokens": round(float(dt.mean()), 1), "tokens_ci95": ct,
                                         "recall": round(float(dr.mean()), 4), "recall_ci95": cr,
                                         "success": bool(ct[1] < 0 and cr[1] >= 0)}
        tok[f"recall_{target:g}"] = row
    rep["tokens_at_target"] = tok
    rep["primary_success"] = tests.get("M1-iter_vs_M1_recall@5", {}).get("success")
    (HERE / "results" / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
