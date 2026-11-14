"""Fine-tune A1 on MuSiQue train (PROTOCOL.md). Resumable; checkpoints every --save-every steps.

  uv run python train.py --mode iter --tag m1_iter      # all steps t = 0..h-1
  uv run python train.py --mode step0 --tag m1          # step-0 examples only (same draw)
  uv run python train.py --mode iter --tag smoke --smoke 40   # throughput test, nothing kept
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import chain, encode, load, paragraphs_of, query

HERE = Path(__file__).resolve().parent
A1 = HERE.parent / "exp04_astra_edit" / "models" / "a1"
torch.set_num_threads(8)


def examples(rows, mode, n_neg=6):
    """(query, paragraph, label). Negatives are seeded per (question, step), so step0 = iter's step-0 subset."""
    out = []
    for qi, r in enumerate(rows):
        _, paras = paragraphs_of(r)
        ch = chain(r)
        non = [i for i in range(len(paras)) if i not in ch]
        for t in range(len(ch) if mode == "iter" else 1):
            rng = np.random.default_rng([qi, t])
            q = query(r["question"], [paras[i] for i in ch[:t]])
            out += [(q, paras[i], 1.0) for i in ch[t:]]
            out += [(q, paras[i], 0.0) for i in rng.choice(non, n_neg, replace=False)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("iter", "step0"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-questions", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--smoke", type=int, default=0, help="run this many steps, report throughput, save nothing")
    args = ap.parse_args()
    torch.manual_seed(0)

    rows = load("train")
    # seeded random subset (deviation 1 in results/budget.json: the train ids are sorted, so "first N" = 2-hop only)
    rows = [rows[i] for i in sorted(np.random.default_rng(9).permutation(len(rows))[: args.n_questions])]
    pairs = examples(rows, args.mode)
    out = HERE / "models" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    print(f"mode={args.mode} questions={len(rows)} pairs={len(pairs)} pos={sum(p[2] for p in pairs):.0f}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(A1))
    model = AutoModelForSequenceClassification.from_pretrained(str(A1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = math.ceil(len(pairs) / args.batch)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    step, secs = 0, 0.0
    if ckpt.exists() and not args.smoke:
        st = torch.load(ckpt)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, secs = st["step"], st["secs"]
        print(f"resumed at step {step}/{total}", flush=True)
    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)
    model.train()
    t0 = time.time()
    while step < total:
        idx = order[step * args.batch:(step + 1) * args.batch]
        q, p, y = zip(*(pairs[i] for i in idx))
        enc = encode(tok, list(q), list(p), args.max_len)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**enc).logits.squeeze(-1), torch.tensor(y))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 50 == 0 or (args.smoke and step % 10 == 0):
            el = secs + time.time() - t0
            print(f"step {step}/{total} loss {loss.item():.4f} {el / 60:.1f} min, eta {el / step * (total - step) / 60:.0f} min",
                  flush=True)
        if args.smoke and step >= args.smoke:
            rate = (time.time() - t0) / step
            print(json.dumps({"sec_per_step": round(rate, 2), "steps": total, "projected_h": round(rate * total / 3600, 2)}))
            return
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "secs": secs + time.time() - t0}, ckpt)
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    (out / "train_info.json").write_text(json.dumps(
        {**vars(args), "questions": len(rows), "pairs": len(pairs), "steps": total,
         "train_minutes": round((secs + time.time() - t0) / 60, 1)}, indent=1))
    ckpt.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
