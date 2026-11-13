"""exp07: chain-aware adaptive evidence selection on top of A1 (no extra cross-encoder passes).

  uv run python chain.py              # train on the exp05 pool, tune on dev, report test once

Inputs: ../exp05_failure_anatomy/cache/a1_{pool,dev,test}.npz (A1 scores + 384-d pooler features
per pair; built by exp05 score_pool.py) and the paragraph texts.

Pair scorer: for the 6 pairs among A1's top 4 paragraphs, predict P(pair == the two gold paragraphs).
  features (pair ordered by A1 score, i above j):
    A1      : s_i, s_j, s_i - s_j, rank_i, rank_j
    text    : title_j in para_i, title_i in para_j, title_i in q, title_j in q,
              Jaccard(p_i, p_j), Jaccard(q, p_i), Jaccard(q, p_j), shared capitalised words(p_i, p_j)
    vectors : h_i + h_j, h_i * h_j, |h_i - h_j|   (A1 pooler outputs)
  models  : "text" (logistic regression on A1 + text), "full" (small MLP on everything)
Policies (paragraph sets per question):
  a1_topk      : A1 top-2 / 3 / 4
  a1_thresh    : A1 score >= t (at least 1), exp02 rule
  pair_best    : the best pair (2 paragraphs)
  adaptive     : best pair if p >= tau2; else best pair + next A1 paragraph if p >= tau3; else A1 top 4
Pre-registered comparison: see PROTOCOL.md.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from math import ceil
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import _load, build_retrieval_prompt, gold_mask, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "exp05_failure_anatomy" / "cache"
WORD = re.compile(r"[a-z0-9]+")
CAP = re.compile(r"\b[A-Z][a-zA-Z0-9]+")
STOP = set("the a an of in on and or to for is was were be by with as at from which what who whom whose when where "
           "how that this these those it its did does do are has have had than both same".split())
torch.manual_seed(0)


def words(s):
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def jac(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


def norm(s):
    return " ".join(WORD.findall(s.lower()))


def contains(text, s):
    s = norm(s)
    return float(bool(s) and f" {s} " in f" {norm(text)} ")


def load_split(split, rows):
    z = dict(np.load(CACHE / f"a1_{split}.npz"))  # load once (NpzFile re-decompresses on every access)
    out = []
    for k, qid in enumerate(z["ids"]):
        r = rows[str(qid)]
        titles, paras = paragraphs_of(r)
        m = z["owner"] == k
        s, h = z["score"][m].astype(np.float32), z["feat"][m].astype(np.float32)
        g = np.array(gold_mask(r), bool)
        if len(s) != len(paras):
            continue
        out.append({"id": str(qid), "q": r["question"], "titles": titles, "paras": paras, "s": s, "h": h, "gold": g,
                    "base": n_tokens(build_retrieval_prompt(r["question"], [])),
                    "ptok": np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)])})
    return out


def pair_feats(x):
    order = np.argsort(-x["s"])[:4]
    rows, pairs = [], []
    for a, b in combinations(range(len(order)), 2):
        i, j = order[a], order[b]
        ti, tj, pi, pj = x["titles"][i], x["titles"][j], x["paras"][i], x["paras"][j]
        ci, cj = set(CAP.findall(pi)) - {ti}, set(CAP.findall(pj)) - {tj}
        txt = [x["s"][i], x["s"][j], x["s"][i] - x["s"][j], a + 1, b + 1,
               contains(pi, tj), contains(pj, ti), contains(x["q"], ti), contains(x["q"], tj),
               jac(pi, pj), jac(x["q"], pi), jac(x["q"], pj), len(ci & cj) / 10]
        vec = np.concatenate([x["h"][i] + x["h"][j], x["h"][i] * x["h"][j], np.abs(x["h"][i] - x["h"][j])])
        rows.append((np.array(txt, np.float32), vec))
        pairs.append((i, j))
    y = np.array([float(x["gold"][i] and x["gold"][j] and x["gold"].sum() == 2) for i, j in pairs], np.float32)
    return rows, pairs, y


def matrices(data):
    T, V, Y, Q = [], [], [], []
    for k, x in enumerate(data):
        rows, pairs, y = pair_feats(x)
        x["pairs"] = pairs
        for (t, v), yy in zip(rows, y):
            T.append(t); V.append(v); Y.append(yy); Q.append(k)
    return np.array(T), np.array(V), np.array(Y), np.array(Q)


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xva, yva, epochs=60):
    m = MLP(Xtr.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt, yt, Xv, yv = map(torch.tensor, (Xtr, ytr, Xva, yva))
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.binary_cross_entropy_with_logits(m(Xv), yv))
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    m.load_state_dict(best_state); m.eval()
    return m, {"val_bce": best, "epochs": ep + 1}


def sel_tokens(x, sel):
    return x["base"] + x["ptok"][list(sel)].sum()


def full(x, sel):
    return bool(x["gold"][list(sel)].sum() == x["gold"].sum())


def policies(x, p):
    """Return per-question candidate sets for each policy family (thresholds applied later)."""
    order = np.argsort(-x["s"])
    b = int(np.argmax(p)); i, j = x["pairs"][b]
    nxt = next(k for k in order if k not in (i, j)) if len(order) > 2 else None
    return {"best_pair": (i, j), "pair_plus1": (i, j, nxt) if nxt is not None else (i, j),
            "top4": tuple(order[:4]), "p_best": float(p[b])}


def evaluate(data, P):
    """Per-question outcomes for every policy; P[k] = pair probabilities of question k."""
    res = []
    for x, p in zip(data, P):
        order = np.argsort(-x["s"])
        pol = policies(x, p)
        r = {"p_best": pol["p_best"], "s": x["s"], "order": order}
        for k in (2, 3, 4):
            r[f"top{k}"] = (full(x, order[:k]), sel_tokens(x, order[:k]))
        for name in ("best_pair", "pair_plus1", "top4"):
            r[name] = (full(x, pol[name]), sel_tokens(x, pol[name]))
            r[name + "_idx"] = [int(i) for i in pol[name]]
        res.append(r)
    return res


def thresh_curve(data):
    """A1 threshold rule: (recall, tokens) for every candidate threshold (exact sweep)."""
    need = np.array([x["s"][x["gold"]].min() for x in data])
    cands = np.unique(np.concatenate([np.sort(need)[::-1], [np.inf]]))
    out = []
    for t in cands[::max(1, len(cands) // 400)]:
        rec, tok = [], []
        for x in data:
            keep = np.where(x["s"] >= t)[0]
            keep = keep if len(keep) else np.argsort(-x["s"])[:1]
            rec.append(full(x, keep)); tok.append(sel_tokens(x, keep))
        out.append((float(t), float(np.mean(rec)), float(np.mean(tok))))
    return out


def adaptive(res, tau2, tau3):
    rec, tok, size = [], [], []
    for r in res:
        key = "best_pair" if r["p_best"] >= tau2 else ("pair_plus1" if r["p_best"] >= tau3 else "top4")
        rec.append(r[key][0]); tok.append(r[key][1]); size.append({"best_pair": 2, "pair_plus1": 3, "top4": 4}[key])
    return float(np.mean(rec)), float(np.mean(tok)), float(np.mean(size))


def best_adaptive(res, target):
    grid = np.unique(np.quantile([r["p_best"] for r in res], np.linspace(0, 1, 41)))
    best = None
    for t2 in grid:
        for t3 in grid[grid <= t2]:
            rec, tok, size = adaptive(res, t2, t3)
            if rec >= target and (best is None or tok < best[1]):
                best = (rec, tok, size, float(t2), float(t3))
    return best


def best_thresh(curve, target):
    ok = [c for c in curve if c[1] >= target]
    return min(ok, key=lambda c: c[2]) if ok else None


def main():
    rows = {r["id"]: dict(r) for r in _load("validation")}
    pool, dev, test = (load_split(s, rows) for s in ("pool", "dev", "test"))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pool)); cut = int(0.8 * len(pool))
    tr, va = [pool[i] for i in idx[:cut]], [pool[i] for i in idx[cut:]]
    Ttr, Vtr, Ytr, _ = matrices(tr); Tva, Vva, Yva, _ = matrices(va)
    Tdv, Vdv, _, Qdv = matrices(dev); Tte, Vte, _, Qte = matrices(test)
    sc_t = StandardScaler().fit(Ttr)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc_t.transform(Ttr), Ytr)
    Xtr, Xva = np.hstack([Ttr, Vtr]), np.hstack([Tva, Vva])
    sc_f = StandardScaler().fit(Xtr)
    # exploratory ablation (added after the pre-registered run): A1 scores and ranks only (txt cols 0-4)
    lr_a = LogisticRegression(max_iter=2000, C=1.0).fit(sc_t.transform(Ttr)[:, :5], Ytr)
    mlp, info = fit_mlp(sc_f.transform(Xtr).astype(np.float32), Ytr, sc_f.transform(Xva).astype(np.float32), Yva)

    def probs(model, T, V, Q, n):
        if model == "a1only":
            p = lr_a.predict_proba(sc_t.transform(T)[:, :5])[:, 1]
        elif model == "text":
            p = lr.predict_proba(sc_t.transform(T))[:, 1]
        else:
            with torch.no_grad():
                p = torch.sigmoid(mlp(torch.tensor(sc_f.transform(np.hstack([T, V])).astype(np.float32)))).numpy()
        return [p[Q == k] for k in range(n)]

    report = {"train_questions": len(tr), "val_questions": len(va), "mlp": info}
    for split, data, T, V, Q in (("dev", dev, Tdv, Vdv, Qdv), ("test", test, Tte, Vte, Qte)):
        rep = {}
        for model in ("text", "full", "a1only"):
            res = evaluate(data, probs(model, T, V, Q, len(data)))
            rep[model] = {"bg2_pair_best": float(np.mean([r["best_pair"][0] for r in res])),
                          "bg2_a1_top2": float(np.mean([r["top2"][0] for r in res])),
                          "tokens_pair_best": float(np.mean([r["best_pair"][1] for r in res])),
                          "tokens_a1_top2": float(np.mean([r["top2"][1] for r in res])),
                          "pair_best_only": int(sum(r["best_pair"][0] and not r["top2"][0] for r in res)),
                          "a1_top2_only": int(sum(r["top2"][0] and not r["best_pair"][0] for r in res))}
            rep[model]["_res"] = res
        rep["a1_curve"] = thresh_curve(data)
        report[split] = rep
    # thresholds tuned on dev, applied once to test
    final = {}
    for target in (0.90, 0.95):
        row = {}
        t = best_thresh(report["dev"]["a1_curve"], target)
        te = [c for c in report["test"]["a1_curve"] if c[0] <= t[0]]
        tt = max(te, key=lambda c: c[0]) if te else report["test"]["a1_curve"][-1]
        row["a1_thresh"] = {"dev": t, "test_recall": tt[1], "test_tokens": tt[2]}
        for model in ("text", "full", "a1only"):
            b = best_adaptive(report["dev"][model]["_res"], target)
            rec, tok, size = adaptive(report["test"][model]["_res"], b[3], b[4])
            row[f"adaptive_{model}"] = {"dev": b, "test_recall": rec, "test_tokens": tok, "test_mean_size": size}
        # oracle-on-test frontier (same procedure for both): tokens at the target on test itself
        row["frontier_test"] = {"a1_thresh": best_thresh(report["test"]["a1_curve"], target),
                                **{f"adaptive_{m}": best_adaptive(report["test"][m]["_res"], target) for m in ("text", "full", "a1only")}}
        final[f"recall_{target:g}"] = row
    # paired statistics on test (dev-tuned thresholds): bootstrap over questions + McNemar for bg2
    from math import comb
    rng_b = np.random.default_rng(1)
    sels = {}
    for target in (0.90, 0.95):
        row = final[f"recall_{target:g}"]
        t = row["a1_thresh"]["dev"][0]
        a1 = []
        for x in test:
            keep = np.where(x["s"] >= t)[0]
            keep = keep if len(keep) else np.argsort(-x["s"])[:1]
            a1.append((full(x, keep), sel_tokens(x, keep)))
            sels.setdefault(f"a1_thresh_{target:g}", {})[x["id"]] = [int(i) for i in keep[np.argsort(-x["s"][keep])]]
        for model in ("text", "full", "a1only"):
            d = row[f"adaptive_{model}"]["dev"]
            ad = []
            for r in report["test"][model]["_res"]:
                key = "best_pair" if r["p_best"] >= d[3] else ("pair_plus1" if r["p_best"] >= d[4] else "top4")
                ad.append(r[key])
                sels.setdefault(f"adaptive_{model}_{target:g}", {})[test[len(ad) - 1]["id"]] = r[key + "_idx"]
            dt = np.array([a[1] - b[1] for a, b in zip(ad, a1)], float)
            dr = np.array([float(a[0]) - float(b[0]) for a, b in zip(ad, a1)])
            bs_t = [dt[rng_b.integers(0, len(dt), len(dt))].mean() for _ in range(2000)]
            bs_r = [dr[rng_b.integers(0, len(dr), len(dr))].mean() for _ in range(2000)]
            row[f"adaptive_{model}"]["paired_vs_a1_thresh"] = {
                "tokens_diff_mean": float(dt.mean()), "tokens_diff_ci95": [float(np.percentile(bs_t, 2.5)), float(np.percentile(bs_t, 97.5))],
                "recall_diff_mean": float(dr.mean()), "recall_diff_ci95": [float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))]}
    for model in ("text", "full", "a1only"):
        b_, c_ = report["test"][model]["pair_best_only"], report["test"][model]["a1_top2_only"]
        n_ = b_ + c_
        report["test"][model]["bg2_mcnemar_p"] = min(1.0, 2 * sum(comb(n_, i) for i in range(min(b_, c_) + 1)) / 2 ** n_) if n_ else 1.0
    for split in ("dev", "test"):
        for model in ("text", "full", "a1only"):
            report[split][model].pop("_res")
        report[split].pop("a1_curve")
    report["policies"] = final
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "test_selections.json").write_text(json.dumps(sels))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "chain_results.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
