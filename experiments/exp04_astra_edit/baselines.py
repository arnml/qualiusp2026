"""Track C baselines: structured pruning of A1 without retraining, at matched FLOPs.

  uv run python baselines.py --eval-n 300          # curves on 300 dev questions
  uv run python baselines.py --eval-n 1000 --fractions 0.8 0.6

Importance is estimated on the same practice set the editor sees (heldval, 200 questions).
  layerdrop : drop the top-k layers (Sajjad et al. 2023; Gromov et al. 2024)
  taylor    : |gradient x mask| importance for heads and |act x grad| for neurons
              (Michel et al. 2019 style; uses gradients, which is fine for a baseline)
  wanda     : ||W_out column|| x ||activation|| for neurons and heads, no gradients (Sun et al. 2024)
  random    : random heads/neurons (control)
Units (heads, neurons) are removed greedily by importance per FLOP until the target encoder
FLOPs fraction is reached. Quality = both-gold@3 on dev questions (never seen by the editor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import editlib as el
from data import gold_mask, load_eval, paragraphs_of
from harness import build_practice

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "a1"
torch.set_num_threads(8)


def flatten(qs):
    q, p, owner = [], [], []
    for k, r in enumerate(qs):
        paras = r["paragraphs"] if "paragraphs" in r else paragraphs_of(r)[1]
        for para in paras:
            q.append(r["question"]); p.append(para); owner.append(k)
    return q, p, np.array(owner)


def bg3(scores, rows, owner):
    ok = []
    for k, r in enumerate(rows):
        g = np.array(r["gold"] if "gold" in r else gold_mask(r))
        s = scores[owner == k]
        ok.append(g[np.argsort(-s)[:3]].sum() == g.sum())
    return float(np.mean(ok))


@torch.no_grad()
def score(model, tok, q, p, bs=64):
    out = []
    for i in range(0, len(q), bs):
        enc = tok(q[i:i + bs], p[i:i + bs], truncation=True, max_length=256, padding=True, return_tensors="pt")
        out.append(model(**enc).logits.squeeze(-1).numpy())
    return np.concatenate(out)


def importances(model, tok, practice):
    """Per-head and per-neuron importance: taylor (needs grads) and wanda (no grads)."""
    q, p, owner = flatten(practice)
    for prm in model.parameters():
        prm.requires_grad_(False)  # only the masks need gradients
    y = torch.tensor([g for r in practice for g in r["gold"]], dtype=torch.float)
    L = len(model.bert.encoder.layer)
    hmask = torch.ones(L, el.N_HEADS, requires_grad=True)
    nmask = torch.ones(L, el.FFN, requires_grad=True)
    ctx_norm = torch.zeros(L, el.N_HEADS)
    act_norm = torch.zeros(L, el.FFN)
    hooks = []
    for l, layer in enumerate(model.bert.encoder.layer):
        def pre(mod, args, l=l):
            x = args[0]
            ctx_norm[l] += x.detach().reshape(*x.shape[:-1], el.N_HEADS, el.HEAD).pow(2).sum((0, 1, 3))
            x = (x.reshape(*x.shape[:-1], el.N_HEADS, el.HEAD) * hmask[l][:, None]).reshape(x.shape)
            return (x,) + args[1:]
        def post(mod, inp, out, l=l):
            act_norm[l] += out.detach().pow(2).sum((0, 1))
            return out * nmask[l]
        hooks.append(layer.attention.output.dense.register_forward_pre_hook(pre))
        hooks.append(layer.intermediate.register_forward_hook(post))
    model.train(False)
    for i in range(0, len(q), 32):
        enc = tok(q[i:i + 32], p[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
        logit = model(**enc).logits.squeeze(-1)
        torch.nn.functional.binary_cross_entropy_with_logits(logit, y[i:i + 32], reduction="sum").backward()
    for h in hooks:
        h.remove()
    sd = model.state_dict()
    wo_head = torch.stack([sd[el.O.format(l=l) + ".weight"].reshape(el.HIDDEN, el.N_HEADS, el.HEAD).pow(2).sum((0, 2)).sqrt() for l in range(L)])
    wo_neur = torch.stack([sd[el.FO.format(l=l) + ".weight"].pow(2).sum(0).sqrt() for l in range(L)])
    return {
        "taylor": (hmask.grad.abs().numpy(), nmask.grad.abs().numpy()),
        "wanda": ((wo_head * ctx_norm.sqrt()).numpy(), (wo_neur * act_norm.sqrt()).numpy()),
    }


def greedy_state(head_imp, neur_imp, frac, rng=None):
    """Remove lowest importance-per-FLOP units until encoder FLOPs <= frac of the original."""
    st = el.EditState("C")
    full = st.cost()["encoder_gflops_per_question"]
    one_head = full - _cost_with(heads={(0, 0)})
    one_neur = full - _cost_with(neurons={(0, 0)})
    units = [(head_imp[l, h] / one_head, "h", l, h) for l in range(el.N_LAYERS) for h in range(el.N_HEADS)]
    units += [(neur_imp[l, i] / one_neur, "n", l, i) for l in range(el.N_LAYERS) for i in range(el.FFN)]
    if rng is not None:
        rng.shuffle(units)
    else:
        units.sort(key=lambda u: u[0])
    target = frac * full
    cur = full
    for _, kind, l, i in units:
        if cur <= target:
            break
        if kind == "h":
            if sum(1 for a, _ in st.heads if a == l) >= el.N_HEADS - 1:
                continue  # keep at least one head per layer
            st.heads.add((l, i)); cur -= one_head
        else:
            if sum(1 for a, _ in st.neurons if a == l) >= el.FFN - 8:
                continue
            st.neurons.add((l, i)); cur -= one_neur
    return st


def _cost_with(heads=(), neurons=()):
    s = el.EditState("C"); s.heads = set(heads); s.neurons = set(neurons)
    return s.cost()["encoder_gflops_per_question"]


def layerdrop_state(frac):
    st = el.EditState("C")
    full = st.cost()["encoder_gflops_per_question"]
    for l in range(el.N_LAYERS - 1, 0, -1):
        if st.cost()["encoder_gflops_per_question"] <= frac * full:
            break
        st.layers.add(l)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--methods", nargs="+", default=["layerdrop", "taylor", "wanda", "random"])
    ap.add_argument("--out", help="output file name (default baselines_dev<N>.json)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    base = el.load_base(MODEL)
    practice = build_practice("heldval", 200)
    imp = importances(el.build_model(MODEL, el.EditState("C"), base), tok, practice)
    dev = load_eval("dev")[: args.eval_n]
    q, p, owner = flatten(dev)
    full = el.EditState("C").cost()["encoder_gflops_per_question"]
    rows = []
    ref = bg3(score(el.build_model(MODEL, el.EditState("C"), base), tok, q, p), dev, owner)
    rows.append({"method": "A1_unedited", "frac": 1.0, "gflops": full, "dev_bg3": ref})
    print(rows[-1], flush=True)
    for frac in args.fractions:
        cands = {"layerdrop": layerdrop_state(frac),
                 "taylor": greedy_state(*imp["taylor"], frac),
                 "wanda": greedy_state(*imp["wanda"], frac),
                 "random": greedy_state(*imp["taylor"], frac, rng=np.random.default_rng(0))}
        for name, st in cands.items():
            if name not in args.methods:
                continue
            g = st.cost()["encoder_gflops_per_question"]
            s = bg3(score(el.build_model(MODEL, st, base), tok, q, p), dev, owner)
            rows.append({"method": name, "frac": frac, "gflops": g, "dev_bg3": s,
                         "heads": len(st.heads), "neurons": len(st.neurons), "layers": sorted(st.layers)})
            print(rows[-1], flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / (args.out or f"baselines_dev{args.eval_n}.json")).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
