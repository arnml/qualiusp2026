"""Fine-tune a small cross-encoder to score (question, paragraph) as gold / not gold.

  python train.py --n-train 0   --tag full     # all ~90k train questions
  python train.py --n-train 200 --tag n200 --epochs 10
  python train.py --n-train 50  --tag n50  --epochs 20

Pointwise BCE over all 10 paragraphs of each question (2 gold, 8 BM25 distractors).
Checkpoints are saved to RESULTS_DIR/<tag>/ every --save-every steps and resumed
automatically, so a Colab disconnect only loses the current interval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import gold_mask, load_train, paragraphs_of, results_dir

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"


def pairs_from(rows):
    out = []
    for r in rows:
        _, paras = paragraphs_of(r)
        for p, y in zip(paras, gold_mask(r), strict=True):
            out.append((r["question"], p, float(y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=0, help="0 = all train questions")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = results_dir() / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"

    rows = load_train(args.n_train or None, seed=args.seed)
    pairs = pairs_from(rows)
    print(f"train questions={len(rows)} pairs={len(pairs)} device={dev}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(pairs) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    step = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed from step {step}/{total}")

    def save():
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "step": step}, ckpt)

    model.train()
    t0 = time.time()
    while step < total:
        epoch = step // steps_per_epoch
        order = list(range(len(pairs)))
        random.Random(args.seed + epoch).shuffle(order)
        for b in range(step % steps_per_epoch, steps_per_epoch):
            idx = order[b * args.batch : (b + 1) * args.batch]
            q, p, y = zip(*(pairs[i] for i in idx), strict=True)
            enc = tok(list(q), list(p), truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), torch.tensor(y, device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                rate = step / (time.time() - t0)
                print(f"step {step}/{total} loss {loss.item():.4f} ({rate:.1f} steps/s)", flush=True)
            if step % args.save_every == 0:
                save()
            if step >= total:
                break

    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "train_args.json").write_text(json.dumps(
        {**vars(args), "base": BASE, "n_questions": len(rows), "n_pairs": len(pairs),
         "steps": total}, indent=1))
    if ckpt.exists():
        ckpt.unlink()  # final model saved; drop optimizer state to save Drive space
    print(f"saved {out / 'model'}")


if __name__ == "__main__":
    main()
