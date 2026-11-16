"""exp13 ranking (PROTOCOL.md): exp12's frozen GPU models, one-shot and iterative inference, on the fresh splits.

  python rank.py --split f4_dev --only iter_iter ctrl_oneshot
Writes RESULTS_DIR/rank_<split>.jsonl: one line per config with per-question orders, one-shot scores (top 10) and passes.
Resumable per config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from data import encode, load, paragraphs_of, query
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MODELS_DIR", HERE / "models"))  # exp12 models (Colab Drive)
A1 = Path(os.environ.get("A1_DIR", HERE.parent / "exp04_astra_edit" / "models" / "a1"))  # tokenizer only
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 4
CONFIGS = {
    "iter_iter": ("m1_iter_s0", True),
    "ctrl_oneshot": ("m1_ctrl", False),
    "iter_iter_s1": ("m1_iter_s1", True),
    "ctrl_oneshot_s1": ("m1_ctrl_s1", False),
    "iter_iter_s2": ("m1_iter_s2", True),
    "ctrl_oneshot_s2": ("m1_ctrl_s2", False),
    "iter_oneshot": ("m1_iter_s0", False),
    "ctrl_iter": ("m1_ctrl", True),
}  # 2x2 on 4-hop (descriptive)


@torch.no_grad()
def rank_all(model, tok, rows, iterative):
    """exp09 inference: one-shot sort, or 4 greedy steps with the tail ordered by the step-3 scores."""

    def score(q, ps):
        return model(**encode(tok, [q] * len(ps), ps).to(DEV)).logits.float().cpu().squeeze(-1).reshape(-1).tolist()

    out = []
    for r in rows:
        _, paras = paragraphs_of(r)
        rest, order, passes, top, qlen = list(range(len(paras))), [], 0, None, []
        for _ in range(STEPS if iterative else 1):
            q = query(r["question"], [paras[i] for i in order])
            qlen.append(len(tok(q)["input_ids"]))
            s = score(q, [paras[i] for i in rest])
            passes += len(rest)
            ranked = sorted(zip(s, rest), key=lambda z: -z[0])
            if not iterative:
                order, top = [i for _, i in ranked], [round(v, 4) for v, _ in ranked[:10]]
                break
            order.append(ranked[0][1])
            rest = [i for _, i in ranked[1:]]
        order = order + (rest if iterative else [])
        assert sorted(order) == list(range(len(paras))), "order is not a full permutation"
        out.append({"order": order, "passes": passes, "top10": top, "max_query_tokens": max(qlen)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--only", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    assert DEV == "cuda" or os.environ.get("ALLOW_CPU"), "PROTOCOL: Colab GPU"
    tok = AutoTokenizer.from_pretrained(str(A1))
    rows = load(args.split)
    out = Path(os.environ.get("RESULTS_DIR", HERE / "results")) / f"rank_{args.split}.jsonl"
    done = (
        {(j["config"], j["sha256"]) for j in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    )
    for name in args.only:
        tag, iterative = CONFIGS[name]
        sha = hashlib.sha256((MODELS / tag / "model" / "model.safetensors").read_bytes()).hexdigest()
        if (name, sha) in done:
            continue
        assert name not in {c for c, _ in done}, f"{name} exists with a different model hash"
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(str(MODELS / tag / "model")).eval().to(DEV)
        res = rank_all(model, tok, rows, iterative)
        with out.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "config": name,
                        "model": tag,
                        "iterative": iterative,
                        "ids": [r["id"] for r in rows],
                        "orders": [x["order"] for x in res],
                        "passes": [x["passes"] for x in res],
                        "top10": [x["top10"] for x in res],
                        "max_query_tokens": [x["max_query_tokens"] for x in res],
                        "sha256": sha,
                        "device": DEV,
                    }
                )
                + "\n"
            )
        print(args.split, name, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
