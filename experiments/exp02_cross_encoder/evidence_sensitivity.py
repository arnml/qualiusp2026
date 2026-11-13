"""Answer F1 split by whether the selector kept both gold paragraphs (exp02 test, no new LLM calls).

  uv run python evidence_sensitivity.py
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from generate import f1_em
from data import load_eval, gold_mask
rows = {r["id"]: r for r in load_eval("test")}
def load(c):
    return {d["id"]: d for d in map(json.loads, open(f"results/generation/{c}.jsonl", encoding="utf-8"))}
G = {c: load(c) for c in ["a1_top2", "a1_t0.02", "k10_all", "oracle_gold"]}
for c in ["a1_top2", "a1_t0.02"]:
    hit, miss = [], []
    for qid, d in G[c].items():
        r = rows[qid]; g = np.array(gold_mask(r)).astype(bool)
        full = g[d["paragraphs"]].sum() == g.sum()
        f = f1_em(d["answer"], r["answer"])[0]
        (hit if full else miss).append((qid, f))
    mf = [f for _, f in miss]; ids = [q for q, _ in miss]
    print(f"{c}: both-gold selected {len(hit)} (F1 {np.mean([f for _,f in hit]):.3f}) | missing a gold {len(miss)} (F1 {np.mean(mf):.3f};"
          f" same questions with all 10: {np.mean([f1_em(G['k10_all'][q]['answer'], rows[q]['answer'])[0] for q in ids]):.3f};"
          f" with oracle gold: {np.mean([f1_em(G['oracle_gold'][q]['answer'], rows[q]['answer'])[0] for q in ids]):.3f})")
