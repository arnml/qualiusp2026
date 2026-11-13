"""Exploratory (not pre-registered): A0 (off-the-shelf MiniLM-L6) vs A1 top-k full recall on MuSiQue test,
plus mean prompt tokens of sending all paragraphs. Writes results/a0_vs_a1_topk_test.json."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import build_retrieval_prompt, gold_mask, load, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)


@torch.no_grad()
def scores(path, rows):
    tok = AutoTokenizer.from_pretrained(str(path)); m = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
    out = []
    for r in rows:
        _, p = paragraphs_of(r)
        enc = tok([r["question"]] * len(p), p, truncation=True, max_length=256, padding=True, return_tensors="pt")
        out.append(m(**enc).logits.squeeze(-1).numpy())
    return out


def curve(sc, rows):
    rank = np.array([np.where(np.array(gold_mask(r), bool)[np.argsort(-s)])[0].max() + 1 for s, r in zip(sc, rows)])
    return {k: round(float((rank <= k).mean()), 3) for k in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)}


rows = load("test")
z = dict(np.load(HERE / "cache" / "a1_test.npz"))
a1 = [z["score"][z["owner"] == k] for k in range(len(rows))]
a0 = scores(HERE.parent / "exp04_astra_edit" / "models" / "a0", rows)
full = np.mean([n_tokens(build_retrieval_prompt(r["question"], paragraphs_of(r)[1])) for r in rows])
rep = {"a1": curve(a1, rows), "a0": curve(a0, rows), "all_paragraphs_tokens": round(float(full), 1)}
(HERE / "results" / "a0_vs_a1_topk_test.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
