"""Score every (question, paragraph) of the dev and test questions with a cross-encoder.

  python score.py --model zeroshot          # off-the-shelf ms-marco MiniLM (arm A0)
  python score.py --model full              # RESULTS_DIR/full/model

Writes RESULTS_DIR/scores_<name>.json : {split: {qid: [10 logits]}}. Small (~0.5 MB),
everything else (selection, tokens, recall) is computed from it by analyze.py.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_eval, paragraphs_of, results_dir
from train import BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'zeroshot' or a tag under RESULTS_DIR")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = BASE if args.model == "zeroshot" else str(results_dir() / args.model / "model")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()

    out = {}
    for split in ("dev", "test"):
        rows = load_eval(split)
        pairs, owner = [], []
        for r in rows:
            _, paras = paragraphs_of(r)
            for p in paras:
                pairs.append((r["question"], p))
                owner.append(r["id"])
        logits = []
        with torch.no_grad():
            for i in range(0, len(pairs), args.batch):
                q, p = zip(*pairs[i : i + args.batch], strict=True)
                enc = tok(list(q), list(p), truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to(dev)
                with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                    logits += model(**enc).logits.squeeze(-1).float().cpu().tolist()
        per_q: dict[str, list[float]] = {}
        for qid, s in zip(owner, logits, strict=True):
            per_q.setdefault(qid, []).append(s)
        out[split] = per_q
        print(f"{split}: scored {len(pairs)} pairs")

    dest = results_dir() / f"scores_{args.model}.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
