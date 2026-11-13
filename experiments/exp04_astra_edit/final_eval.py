"""Stage 3 final checks: test400 both-gold@3, encoder params and CPU latency after physical pruning.

  uv run python final_eval.py --run s3_C_B50 --frac 0.8659

Compares A1, the editor's final edit and Wanda at the given FLOPs fraction. Removed neurons and
heads are sliced out of the weight matrices (not zeroed), and scores are checked against the
zeroed model before timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import editlib as el
from baselines import MODEL, bg3, flatten, greedy_state, importances, score
from data import load_eval
from harness import build_practice

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)


def physical(model, st: el.EditState):
    """Slice removed MLP neurons and attention heads out of the (already zeroed) model."""
    kept_layers = [l for l in range(el.N_LAYERS) if l not in st.layers]
    for new_l, l in enumerate(kept_layers):
        layer = model.bert.encoder.layer[new_l]
        drop_n = {i for a, i in st.neurons if a == l}
        keep_n = torch.tensor([i for i in range(el.FFN) if i not in drop_n], dtype=torch.long)
        inter, out = layer.intermediate.dense, layer.output.dense
        new_in = torch.nn.Linear(el.HIDDEN, len(keep_n))
        new_in.weight.data = inter.weight.data[keep_n].clone(); new_in.bias.data = inter.bias.data[keep_n].clone()
        new_out = torch.nn.Linear(len(keep_n), el.HIDDEN)
        new_out.weight.data = out.weight.data[:, keep_n].clone(); new_out.bias.data = out.bias.data.clone()
        layer.intermediate.dense, layer.output.dense = new_in, new_out
        drop_h = {h for a, h in st.heads if a == l}
        if drop_h:
            keep_h = [h for h in range(el.N_HEADS) if h not in drop_h]
            idx = torch.tensor([h * el.HEAD + j for h in keep_h for j in range(el.HEAD)], dtype=torch.long)
            sa = layer.attention.self
            for name in ("query", "key", "value"):
                lin = getattr(sa, name)
                new = torch.nn.Linear(el.HIDDEN, len(idx))
                new.weight.data = lin.weight.data[idx].clone(); new.bias.data = lin.bias.data[idx].clone()
                setattr(sa, name, new)
            o = layer.attention.output.dense
            new_o = torch.nn.Linear(len(idx), el.HIDDEN)
            new_o.weight.data = o.weight.data[:, idx].clone(); new_o.bias.data = o.bias.data.clone()
            layer.attention.output.dense = new_o
            sa.num_attention_heads = len(keep_h)
            sa.all_head_size = len(idx)
    return model.eval()


def n_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith("bert.encoder"))


@torch.no_grad()
def latency(model, tok, rows, reps=3):
    """Median over reps of the mean CPU ms per question (one question's paragraphs per batch)."""
    batches = []
    for r in rows:
        paras = r["paragraphs"]
        batches.append(tok([r["question"]] * len(paras), paras, truncation=True, max_length=256,
                           padding=True, return_tensors="pt"))
    for enc in batches[:10]:
        model(**enc)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        for enc in batches:
            model(**enc)
        times.append((time.perf_counter() - t) / len(batches) * 1000)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--lat-n", type=int, default=200)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    base = el.load_base(MODEL)
    test = load_eval("test")
    q, p, owner = flatten(test)
    dev = load_eval("dev")[: args.lat_n]
    from data import paragraphs_of
    for r in dev:
        if "paragraphs" not in r:
            r["paragraphs"] = paragraphs_of(r)[1]
    imp = importances(el.build_model(MODEL, el.EditState("C"), base), tok, build_practice("heldval", 200))
    states = {"A1": el.EditState("C"),
              "editor": el.EditState.load(HERE / "runs" / args.run / "work" / "edited", "C"),
              "wanda": greedy_state(*imp["wanda"], args.frac)}
    res = {"run": args.run, "frac": args.frac, "threads": torch.get_num_threads(), "rows": []}
    for name, st in states.items():
        zeroed = el.build_model(MODEL, st, base)
        s_zero = score(zeroed, tok, q, p)
        pruned = physical(el.build_model(MODEL, st, base), st)
        s_phys = score(pruned, tok, q, p)
        row = {"model": name, "gflops": st.cost()["encoder_gflops_per_question"],
               "params_formula": st.cost()["encoder_params"], "params_physical": n_params(pruned),
               "max_abs_diff_phys_vs_zeroed": float(np.abs(s_zero - s_phys).max()),
               "test400_bg3": bg3(s_phys, test, owner),
               "cpu_ms_per_question": latency(pruned, tok, dev)}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    (HERE / "runs" / args.run / "final_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
