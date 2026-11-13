"""Shared helpers for exp06: model loading, scoring and selection metrics (exp02/exp04 protocol)."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import build_retrieval_prompt, gold_mask, load_eval, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "exp04_astra_edit" / "models"  # git-ignored; see exp04 README
A0_DIR, A1_DIR = MODELS / "a0", MODELS / "a1"
torch.set_num_threads(8)


def state(d: Path) -> dict[str, torch.Tensor]:
    return {k: v.float() for k, v in load_file(str(d / "model.safetensors")).items()}


def build(sd: dict[str, torch.Tensor]):
    m = AutoModelForSequenceClassification.from_pretrained(str(A0_DIR))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("position_ids" in k for k in missing), missing
    return m.eval()


def prep(split: str, n: int | None = None):
    rows = load_eval(split)[:n] if n else load_eval(split)
    items = []
    for r in rows:
        _, paras = paragraphs_of(r)
        items.append({"id": r["id"], "question": r["question"], "paragraphs": paras,
                      "gold": np.array(gold_mask(r), bool),
                      "base": n_tokens(build_retrieval_prompt(r["question"], [])),
                      "ptok": np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)])})
    return items


_TOK = None


@torch.no_grad()
def score(model, items, bs: int = 64) -> dict[str, np.ndarray]:
    global _TOK
    _TOK = _TOK or AutoTokenizer.from_pretrained(str(A0_DIR))
    q, p, owner = [], [], []
    for k, it in enumerate(items):
        for para in it["paragraphs"]:
            q.append(it["question"]); p.append(para); owner.append(k)
    owner = np.array(owner)
    out = []
    for i in range(0, len(q), bs):
        enc = _TOK(q[i:i + bs], p[i:i + bs], truncation=True, max_length=256, padding=True, return_tensors="pt")
        out.append(model(**enc).logits.squeeze(-1).numpy())
    s = np.concatenate(out)
    return {it["id"]: s[owner == k] for k, it in enumerate(items)}


def metrics(items, S, target: float = 0.9) -> dict:
    """bg3, and tok@90 with the threshold set on these items (exact sweep on raw scores)."""
    need = np.array([S[it["id"]][it["gold"]].min() for it in items])
    t = np.sort(need)[::-1][ceil(target * len(need)) - 1]
    tok, top3 = [], []
    for it in items:
        s = S[it["id"]]
        keep = np.where(s >= t)[0]
        if not len(keep):
            keep = np.argsort(-s)[:1]
        tok.append(it["base"] + it["ptok"][keep].sum())
        top3.append(it["gold"][np.argsort(-s)[:3]].sum() == it["gold"].sum())
    return {"bg3": float(np.mean(top3)), "tok90": float(np.mean(tok))}
