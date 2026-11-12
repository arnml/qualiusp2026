"""Go/no-go probe for exp03 (our diagnostic; NOT something the editor is allowed to do).

Question: does the off-the-shelf cross-encoder (A0) already carry a linear signal that
separates gold from distractor paragraphs inside its layers, a signal its output does
not use? If yes, a targeted weight edit is plausible.

Data: practice = the same 200 train questions the editor / A2 get (load_train(200, seed=0)).
      held-out = 300 dev questions (never shown to the editor).
Per layer (embeddings, layers 1..6, pooler) we measure pair-level AUC (gold vs distractor)
and question-level 'both gold in top 3' of
  (a) a mean-difference direction  mu_gold - mu_distr   <- an ALLOWED editor operation
  (b) a logistic-regression probe                        <- upper bound, NOT allowed to editors
and compare with A0's own score. We also test the lexical-overlap hypothesis.

  uv run python probe.py
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import gold_mask, load_eval, load_train, paragraphs_of, results_dir

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
N_DEV = 300
torch.set_num_threads(8)


def pairs_of(rows):
    q, p, y, qid = [], [], [], []
    for k, r in enumerate(rows):
        _, paras = paragraphs_of(r)
        for para, g in zip(paras, gold_mask(r), strict=True):
            q.append(r["question"]); p.append(para); y.append(g); qid.append(k)
    return q, p, np.array(y), np.array(qid)


@torch.no_grad()
def extract(model, tok, q, p, batch=32):
    """[CLS] after embeddings + each of 6 layers, pooled output, and the score."""
    cls, pooled, score = [], [], []
    for i in range(0, len(q), batch):
        enc = tok(q[i:i + batch], p[i:i + batch], truncation=True, max_length=256,
                  padding=True, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack([h[:, 0] for h in out.hidden_states], 1)  # (b, 7, 384)
        cls.append(hs.numpy())
        pooled.append(model.bert.pooler(out.hidden_states[-1]).numpy())
        score.append(out.logits.squeeze(-1).numpy())
        if (i // batch) % 20 == 0:
            print(f"  {i}/{len(q)}", flush=True)
    return np.concatenate(cls), np.concatenate(pooled), np.concatenate(score)


def top3_both(score, y, qid):
    ok = []
    for k in np.unique(qid):
        m = qid == k
        top = np.argsort(-score[m])[:3]
        ok.append(y[m][top].sum() == y[m].sum())
    return float(np.mean(ok))


def jaccard(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE).eval()

    tr_rows = load_train(200, seed=0)
    dv_rows = load_eval("dev")[:N_DEV]
    trq, trp, ytr, qtr = pairs_of(tr_rows)
    dvq, dvp, ydv, qdv = pairs_of(dv_rows)
    print(f"practice pairs {len(ytr)}, held-out pairs {len(ydv)}")
    Htr, Ptr, Str = extract(model, tok, trq, trp)
    Hdv, Pdv, Sdv = extract(model, tok, dvq, dvp)

    res = {"A0_output": {"auc": roc_auc_score(ydv, Sdv), "top3_both": top3_both(Sdv, ydv, qdv)}}
    feats = {f"layer{l}": (Htr[:, l], Hdv[:, l]) for l in range(7)}
    feats["pooler"] = (Ptr, Pdv)
    for name, (a, b) in feats.items():
        mu = a[ytr == 1].mean(0) - a[ytr == 0].mean(0)       # allowed: mean difference
        s_md = b @ mu
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(a, ytr)  # upper bound only
        s_lr = clf.decision_function(b)
        res[name] = {
            "meandiff_auc": roc_auc_score(ydv, s_md), "meandiff_top3": top3_both(s_md, ydv, qdv),
            "probe_auc": roc_auc_score(ydv, s_lr), "probe_top3": top3_both(s_lr, ydv, qdv),
            # how aligned is the mean-difference direction with what A0 already reads out?
            "corr_meandiff_vs_A0score": float(np.corrcoef(s_md, Sdv)[0, 1]),
        }

    # Lexical-overlap hypothesis on held-out pairs
    ov = np.array([jaccard(a, b) for a, b in zip(dvq, dvp, strict=True)])
    wrong_d = []  # overlap of distractors that outrank a gold paragraph in A0
    for k in np.unique(qdv):
        m = np.where(qdv == k)[0]
        gmin = Sdv[m][ydv[m] == 1].min()
        wrong_d += [ov[i] for i in m if ydv[i] == 0 and Sdv[i] > gmin]
    res["lexical"] = {
        "overlap_auc_gold_vs_distr": roc_auc_score(ydv, ov),
        "corr_A0score_overlap_distractors": float(np.corrcoef(Sdv[ydv == 0], ov[ydv == 0])[0, 1]),
        "corr_A0score_overlap_gold": float(np.corrcoef(Sdv[ydv == 1], ov[ydv == 1])[0, 1]),
        "mean_overlap_gold": float(ov[ydv == 1].mean()),
        "mean_overlap_distr": float(ov[ydv == 0].mean()),
        "mean_overlap_distr_outranking_gold": float(np.mean(wrong_d)) if wrong_d else None,
    }
    out = results_dir() / "probe.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
