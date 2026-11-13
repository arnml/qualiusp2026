"""Score MuSiQue (dev, test, pool) with A1 once; cache scores + 384-d pooler vectors (float16).

  uv run python score.py        # writes cache/a1_<split>.npz (git-ignored); skips splits already cached
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import gold_mask, load, paragraphs_of

HERE = Path(__file__).resolve().parent
A1 = HERE.parent / "exp04_astra_edit" / "models" / "a1"
torch.set_num_threads(8)


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(str(A1))
    model = AutoModelForSequenceClassification.from_pretrained(str(A1)).eval()
    feats = []
    model.classifier.register_forward_hook(lambda m, i, o: feats.append(i[0].detach().half().numpy()))
    (HERE / "cache").mkdir(exist_ok=True)
    for split in ("test", "dev", "pool"):
        dest = HERE / "cache" / f"a1_{split}.npz"
        if dest.exists():
            continue
        rows = load(split)
        q, p, owner, gold = [], [], [], []
        for k, r in enumerate(rows):
            _, paras = paragraphs_of(r)
            for para, gi in zip(paras, gold_mask(r)):
                q.append(r["question"]); p.append(para); owner.append(k); gold.append(gi)
        feats.clear(); scores = []
        t = time.time()
        for i in range(0, len(q), 64):
            enc = tok(q[i:i + 64], p[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt")
            scores.append(model(**enc).logits.squeeze(-1).numpy())
            if i % 6400 == 0:
                print(split, i, len(q), round(time.time() - t), flush=True)
        np.savez_compressed(dest, ids=np.array([r["id"] for r in rows]), owner=np.array(owner),
                            gold=np.array(gold, np.int8), score=np.concatenate(scores), feat=np.concatenate(feats))
        (HERE / "cache" / f"a1_{split}_meta.json").write_text(json.dumps(
            {"questions": len(rows), "pairs": len(q), "seconds": round(time.time() - t)}))
        print(split, "done", len(rows), "questions", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
