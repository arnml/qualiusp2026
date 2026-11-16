"""exp16 training (PROTOCOL.md): from A1, an iterative model (query = question + gold prefix) and its pair-matched
control (identical pairs and order, query = question only), as in exp12.

  python train.py --ds tw --mode iter --tag tw_iter
  python train.py --ds tw --mode ctrl --tag tw_ctrl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from data import encode, load, query
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
A1 = Path(os.environ.get("A1_DIR", HERE.parent / "exp04_astra_edit" / "models" / "a1"))
MODELS = Path(os.environ.get("MODELS_DIR", HERE / "models"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_NEG = 6


def examples(rows, mode):
    out = []
    for qi, r in enumerate(rows):
        ch, paras = r["chain"], r["paras"]
        non = [i for i in range(len(paras)) if i not in ch]
        for t in range(len(ch)):
            rng = np.random.default_rng([qi, t])
            q = r["question"] if mode == "ctrl" else query(r["question"], [paras[i] for i in ch[:t]])
            out += [(q, paras[i], 1.0) for i in ch[t:]]
            out += [(q, paras[i], 0.0) for i in rng.choice(non, min(N_NEG, len(non)), replace=False)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=("hp", "tw"), required=True)
    ap.add_argument("--mode", choices=("iter", "ctrl"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert DEV == "cuda" or os.environ.get("ALLOW_CPU"), "PROTOCOL: Colab GPU"
    torch.manual_seed(args.seed)
    rows = load(f"{args.ds}_train")
    pairs = examples(rows, args.mode)
    if args.mode == "ctrl":
        ref = examples(rows, "iter")
        assert len(ref) == len(pairs) and all(a[1:] == b[1:] for a, b in zip(ref, pairs)), (
            "ctrl pairs do not match iter"
        )
    out = MODELS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model" / "model.safetensors").exists():
        print("exists:", args.tag)
        return
    tok = AutoTokenizer.from_pretrained(str(A1))
    model = AutoModelForSequenceClassification.from_pretrained(str(A1)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total = math.ceil(len(pairs) / 16)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
    )
    order = list(range(len(pairs)))
    random.Random(args.seed).shuffle(order)
    model.train()
    t0 = time.time()
    for step in range(total):
        idx = order[step * 16 : (step + 1) * 16]
        q, p, y = zip(*(pairs[i] for i in idx))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(**encode(tok, list(q), list(p)).to(DEV)).logits.squeeze(-1), torch.tensor(y, device=DEV)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0:
            print(
                f"{args.tag} step {step + 1}/{total} loss {loss.item():.4f} {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "train_info.json").write_text(
        json.dumps(
            {
                **vars(args),
                "questions": len(rows),
                "pairs": len(pairs),
                "steps": total,
                "train_minutes": round((time.time() - t0) / 60, 1),
                "device": DEV,
            },
            indent=1,
        )
    )
    print("done", args.tag, len(pairs), "pairs", flush=True)


if __name__ == "__main__":
    main()
