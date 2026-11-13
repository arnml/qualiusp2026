"""Track R evaluation: an edited A0 vs A0 (unedited), A2 (A0 fine-tuned on the same 200 practice
questions) and A1 (full fine-tune), on held-out questions the editor never saw.

  uv run python evaluate_r.py --refs              # reference rows only (from exp02 score files)
  uv run python evaluate_r.py --run s1_R_B5       # + the run's final edit, dev1000
  uv run python evaluate_r.py --run s3_R_B50 --test   # + test400 (stage 3 only)

Metrics (exp02 protocol, tokens = the paper's prompt, tiktoken o200k_base):
  bg3      both gold paragraphs in the top 3
  tok@90   mean prompt tokens when keeping every paragraph with score >= t (at least 1), with t the
           highest threshold that keeps both gold for >= 90% of questions. On dev it is set on dev
           itself (same procedure for every model); with --test it is set on dev, applied to test.
Exact threshold sweep on raw scores, so any monotone rescaling of the scores changes nothing.
Paired statistics: exact McNemar on bg3, bootstrap (2,000 resamples, thresholds re-set per resample)
95% CI on the tok@90 difference.
"""

from __future__ import annotations

import argparse
import json
from math import ceil, comb
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import editlib as el
from baselines import flatten, score
from data import build_retrieval_prompt, gold_mask, load_eval, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
A0 = HERE / "models" / "a0"
EXP02 = HERE.parent / "exp02_cross_encoder" / "results"
REFS = {"A0": "scores_zeroshot.json", "A2_n200": "scores_n200.json", "A1": "scores_full.json"}
torch.set_num_threads(8)


def prep(split):
    rows = load_eval(split)
    items = []
    for r in rows:
        _, paras = paragraphs_of(r)
        items.append({"id": r["id"], "gold": np.array(gold_mask(r), bool),
                      "base": n_tokens(build_retrieval_prompt(r["question"], [])),
                      "ptok": np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)])})
    return rows, items


def need(items, S):
    """Per question: the highest threshold that still keeps both gold (= min gold score)."""
    return np.array([S[it["id"]][it["gold"]].min() for it in items])


def thr90(tq):
    return np.sort(tq)[::-1][ceil(0.9 * len(tq)) - 1]


def per_q(items, S, t):
    tok, full, top3 = [], [], []
    for it in items:
        s = S[it["id"]]
        keep = np.where(s >= t)[0]
        if not len(keep):
            keep = np.argsort(-s)[:1]
        tok.append(it["base"] + it["ptok"][keep].sum())
        full.append(it["gold"][keep].sum() == it["gold"].sum())
        top3.append(it["gold"][np.argsort(-s)[:3]].sum() == it["gold"].sum())
    return np.array(tok, float), np.array(full), np.array(top3)


def summary(items, S, t):
    tok, full, top3 = per_q(items, S, t)
    return {"bg3": float(top3.mean()), "tok90": float(tok.mean()), "full_recall": float(full.mean()),
            "threshold": float(t)}


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return {"a_only": x, "b_only": y, "p": p}


def boot_tok_diff(items, Sa, Sb, reps=2000, seed=0):
    """95% CI of tok@90(a) - tok@90(b) on dev, thresholds re-set inside each resample."""
    rng = np.random.default_rng(seed)
    ta, tb = need(items, Sa), need(items, Sb)
    # per-question tokens as a function of threshold are needed per resample; precompute sorted scores
    diffs = []
    n = len(items)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sub = [items[i] for i in idx]
        da = per_q(sub, Sa, thr90(ta[idx]))[0].mean()
        db = per_q(sub, Sb, thr90(tb[idx]))[0].mean()
        diffs.append(da - db)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def load_ref(name):
    d = json.loads((EXP02 / REFS[name]).read_text())
    return {split: {k: np.array(v) for k, v in d[split].items()} for split in d}


def score_edit(run, rows):
    cfg = json.loads((run / "config.json").read_text())
    model_dir = Path(cfg["model_dir"])
    base = el.load_base(model_dir)
    st = el.EditState.load(run / "work" / "edited", cfg["track"])
    errs = st.check(base)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    q, p, owner = flatten(rows)
    s = score(el.build_model(model_dir, st, base), tok, q, p)
    return {r["id"]: s[owner == k] for k, r in enumerate(rows)}, errs, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    dev_rows, dev = prep("dev")
    S = {name: load_ref(name) for name in REFS}
    out = {"dev1000": {}, "test400": {}}
    names = list(REFS)
    if args.run:
        run = HERE / "runs" / args.run
        sd, errs, st = score_edit(run, dev_rows)
        S["edit"] = {"dev": sd}
        out |= {"run": args.run, "valid": not errs, "errors": errs,
                "changed_entries": sum(int((t != el.load_base(Path(json.loads((run / 'config.json').read_text())['model_dir']))[n]).sum())
                                       for n, t in st.raw.items())}
        names.append("edit")
    for name in names:
        out["dev1000"][name] = summary(dev, S[name]["dev"], thr90(need(dev, S[name]["dev"])))
        print(name, out["dev1000"][name], flush=True)
    if args.run:
        _, _, e3 = per_q(dev, S["edit"]["dev"], np.inf)
        out["paired_dev1000"] = {}
        for ref in ("A0", "A2_n200"):
            _, _, r3 = per_q(dev, S[ref]["dev"], np.inf)
            out["paired_dev1000"][ref] = {"bg3_mcnemar_edit_vs_ref": mcnemar(e3, r3),
                                          "tok90_diff_ci95_edit_minus_ref": boot_tok_diff(dev, S["edit"]["dev"], S[ref]["dev"])}
            print(ref, out["paired_dev1000"][ref], flush=True)
    if args.test:
        test_rows, test = prep("test")
        if args.run:
            S["edit"]["test"], _, _ = score_edit(HERE / "runs" / args.run, test_rows)
        for name in names:
            t = thr90(need(dev, S[name]["dev"]))
            out["test400"][name] = summary(test, S[name]["test"], t)
            print("test", name, out["test400"][name], flush=True)
    dest = (HERE / "runs" / args.run / "eval_r.json") if args.run else (HERE / "results" / "trackR_refs.json")
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
