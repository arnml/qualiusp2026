"""Selection metrics from score files (no GPU, no LLM). Same protocol as exp01.

  python analyze.py                      # every results/scores_*.json

For each scorer:
  - top-k (k=1..5): both-gold rate ("full recall"), precision, tokens
  - adaptive: keep paragraphs with sigmoid(score) >= t (at least 1); t chosen on DEV
    for full-recall targets 0.90 / 0.95, then reported once on TEST.
Tokens = the paper's exact prompt, tiktoken o200k_base.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data import build_retrieval_prompt, gold_mask, load_eval, n_tokens, paragraphs_of, results_dir


def prep(split):
    items = {}
    for r in load_eval(split):
        _, paras = paragraphs_of(r)
        items[r["id"]] = {
            "gold": np.array(gold_mask(r), bool),
            "base": n_tokens(build_retrieval_prompt(r["question"], [])),
            "ptok": np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)]),
        }
    return items


def metrics(items, scores, select):
    full, prec, npar, tok = [], [], [], []
    for qid, it in items.items():
        s = np.asarray(scores[qid])
        sel = select(s)
        hit = it["gold"][sel].sum()
        full.append(hit == it["gold"].sum())
        prec.append(hit / len(sel))
        npar.append(len(sel))
        tok.append(it["base"] + it["ptok"][sel].sum())
    return {"full_recall": np.mean(full), "precision": np.mean(prec),
            "n_paras": np.mean(npar), "tokens": np.mean(tok)}


def topk(k):
    return lambda s: np.argsort(-s)[:k]


def thresh(t):
    def f(s):
        p = 1 / (1 + np.exp(-s))
        keep = np.where(p >= t)[0]
        return keep[np.argsort(-s[keep])] if len(keep) else np.argsort(-s)[:1]
    return f


def main():
    rdir = results_dir()
    files = sorted(rdir.glob("scores_*.json"))
    if not files:
        raise SystemExit(f"no scores_*.json in {rdir}")
    items = {"dev": prep("dev"), "test": prep("test")}
    rows = []
    for f in files:
        name = f.stem.removeprefix("scores_")
        sc = json.loads(f.read_text())
        for k in range(1, 6):
            rows.append({"scorer": name, "rule": f"top{k}",
                         **{f"test_{m}": v for m, v in metrics(items["test"], sc["test"], topk(k)).items()}})
        grid = np.round(np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.96, 0.025)]), 3)
        dev = pd.DataFrame([{"t": t, **metrics(items["dev"], sc["dev"], thresh(t))} for t in grid])
        dev.to_csv(rdir / f"dev_thresholds_{name}.csv", index=False)
        for target in (0.90, 0.95):
            ok = dev[dev.full_recall >= target]
            if not len(ok):
                continue
            t = float(ok.sort_values("tokens").iloc[0].t)
            rows.append({"scorer": name, "rule": f"thresh@{target} (t={t})",
                         **{f"test_{m}": v for m, v in metrics(items["test"], sc["test"], thresh(t)).items()}})
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nReference (exp01, test): k=10 1442 tok; Noise-Gate tau=0.20 0.928/1221 tok;"
          " hop-gate 0.918/1089 tok; gold-only 323 tok")


if __name__ == "__main__":
    main()
