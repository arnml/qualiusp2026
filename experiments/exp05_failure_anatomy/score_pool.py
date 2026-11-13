"""Score HotpotQA validation questions with A1 and cache what exp05 and exp07 need.

  uv run python score_pool.py            # pool (validation minus dev/test), dev, test

For every (question, paragraph) pair: A1's score and the 384-d pooler output the classifier reads
(float16). Saved to cache/a1_<split>.npz (git-ignored, rebuildable). The pool is a random sample
(seed 0) of 4,000 validation questions not in dev (1000) or test (400); A1 never saw validation questions in training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import SPLIT_IDS, _load, gold_mask, paragraphs_of

HERE = Path(__file__).resolve().parent
A1 = HERE.parent / "exp04_astra_edit" / "models" / "a1"
torch.set_num_threads(8)
POOL_N = 4000  # random sample of the unused validation questions (seed 0), for CPU time


def rows_for(split):
    ds = _load("validation")
    if split == "pool":
        used = set(SPLIT_IDS["dev"]) | set(SPLIT_IDS["test"])
        pool = [i for i, qid in enumerate(ds["id"]) if qid not in used]
        pick = sorted(np.random.default_rng(0).choice(pool, min(POOL_N, len(pool)), replace=False))
        return [dict(ds[int(i)]) for i in pick]
    pos = {i: k for k, i in enumerate(SPLIT_IDS[split])}
    out = [None] * len(pos)
    for r in ds:
        if r["id"] in pos:
            out[pos[r["id"]]] = dict(r)
    return out


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(str(A1))
    model = AutoModelForSequenceClassification.from_pretrained(str(A1)).eval()
    feats = []
    model.classifier.register_forward_hook(lambda m, i, o: feats.append(i[0].detach().half().numpy()))
    (HERE / "cache").mkdir(exist_ok=True)
    for split in ("dev", "test", "pool"):
        dest = HERE / "cache" / f"a1_{split}.npz"
        if dest.exists():
            continue
        rows = rows_for(split)
        q, p, owner, gold = [], [], [], []
        for k, r in enumerate(rows):
            _, paras = paragraphs_of(r)
            g = gold_mask(r)
            for para, gi in zip(paras, g):
                q.append(r["question"]); p.append(para); owner.append(k); gold.append(gi)
        feats.clear(); scores = []
        t = time.time()
        for i in range(0, len(q), 64):
            enc = tok(q[i:i + 64], p[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt")
            scores.append(model(**enc).logits.squeeze(-1).numpy())
            if i % 6400 == 0:
                print(split, i, len(q), round(time.time() - t), flush=True)
        np.savez_compressed(dest, ids=np.array([r["id"] for r in rows]), owner=np.array(owner),
                            gold=np.array(gold, np.int8), score=np.concatenate(scores),
                            feat=np.concatenate(feats))
        (HERE / "cache" / f"a1_{split}_meta.json").write_text(json.dumps(
            {"questions": len(rows), "pairs": len(q), "seconds": round(time.time() - t)}))
        print(split, "done", len(rows), "questions", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
