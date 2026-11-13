"""exp08: the exp07 adaptive selector on MuSiQue-Ans (see PROTOCOL.md, pre-registered 0d1aa07).

  uv run python select_adaptive.py      # needs cache/a1_{pool,dev,test}.npz from score.py

Stopping scorers predict p_k = P(A1's top k holds all supporting paragraphs), k = 2..10.
Policy `adaptive`: top k for the smallest k with p_k >= tau, else k = 10. tau / the A1 threshold t are tuned on
dev for full-recall targets 0.90 / 0.95 and applied once to test. Writes results/results.json and
results/test_selections.json.
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import build_retrieval_prompt, gold_mask, hops, load, n_tokens, paragraphs_of

HERE = Path(__file__).resolve().parent
KS = list(range(2, 11))  # pre-registered; --kmax overrides (exploratory)
TOPN = 10
TARGETS = (0.90, 0.95)
WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on and or to for is was were be by with as at from which what who whom whose when where "
           "how that this these those it its did does do are has have had than both same".split())
torch.manual_seed(0)


def norm(s):
    return " ".join(WORD.findall(s.lower()))


def words(s):
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def jac(a: set, b: set):
    return len(a & b) / max(1, len(a | b))


def load_split(split):
    z = dict(np.load(HERE / "cache" / f"a1_{split}.npz"))
    rows = load(split)
    out = []
    for k, r in enumerate(rows):
        assert str(z["ids"][k]) == r["id"]
        m = z["owner"] == k
        titles, paras = paragraphs_of(r)
        s, h = z["score"][m].astype(np.float32), z["feat"][m].astype(np.float32)
        g = np.array(gold_mask(r), bool)
        np_ = [f" {norm(p)} " for p in paras]
        nt = [norm(t) for t in titles]
        # M[i, j] = paragraph i mentions the title of paragraph j (i != j)
        M = np.array([[bool(nt[j]) and i != j and f" {nt[j]} " in np_[i] for j in range(len(paras))]
                      for i in range(len(paras))])
        qn = f" {norm(r['question'])} "
        out.append({"id": r["id"], "q": r["question"], "answer": r["answer"], "aliases": r.get("answer_aliases", []),
                    "hops": hops(r), "s": s, "h": h, "gold": g, "M": M,
                    "tq": np.array([bool(t) and f" {t} " in qn for t in nt]),
                    "wq": words(r["question"]), "wp": [words(p) for p in paras],
                    "base": n_tokens(build_retrieval_prompt(r["question"], [])),
                    "ptok": np.array([n_tokens(f"[{i + 1}] {p}\n\n") for i, p in enumerate(paras)])})
    return out


def feats(x):
    """Per k in KS: (a1only, link, vec) features and the label."""
    o = np.argsort(-x["s"]); s = x["s"][o]
    top10 = np.pad(s[:TOPN], (0, max(0, TOPN - len(s))), constant_values=s[-1])
    A, L, V, Y = [], [], [], []
    for k in KS:
        T, O = o[:k], o[k:]
        nxt = o[k] if k < len(o) else None
        kk = min(k, len(s))  # some questions have < 20 paragraphs
        A.append(np.r_[k, top10, s[kk - 1] - (s[kk] if kk < len(s) else s[kk - 1] - 1)])
        sub = x["M"][np.ix_(T, T)]
        dang = x["M"][np.ix_(T, O)].any() if len(O) else False
        dnext = x["M"][T, nxt].any() if nxt is not None else False
        union = set().union(*(x["wp"][i] for i in T))
        L.append([sub.any(axis=0).sum(), float(dang), float(dnext), x["tq"][T].sum(),
                  jac(x["wq"], union), jac(x["wq"], x["wp"][nxt]) if nxt is not None else 0.0])
        hT = x["h"][T]
        V.append(np.r_[hT.mean(0), hT.max(0), x["h"][nxt] if nxt is not None else np.zeros(x["h"].shape[1])])
        Y.append(float(x["gold"][T].sum() == x["gold"].sum()))
    return np.array(A, np.float32), np.array(L, np.float32), np.array(V, np.float32), np.array(Y, np.float32)


def matrices(data):
    A, L, V, Y = zip(*(feats(x) for x in data))
    return np.vstack(A), np.vstack(L), np.vstack(V), np.concatenate(Y)


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))

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


def topk_tables(data):
    """R[q, k-2] = full recall of top k, TOK[q, k-2] = prompt tokens of top k (k in KS)."""
    R = np.zeros((len(data), len(KS)), bool); TOK = np.zeros((len(data), len(KS)))
    for n, x in enumerate(data):
        o = np.argsort(-x["s"])
        for c, k in enumerate(KS):
            R[n, c] = x["gold"][o[:k]].sum() == x["gold"].sum(); TOK[n, c] = x["base"] + x["ptok"][o[:k]].sum()
    return R, TOK


def adaptive_pick(P, tau):
    ok = P >= tau
    return np.where(ok.any(1), ok.argmax(1), len(KS) - 1)


def tune_adaptive(P, R, TOK, target):
    best = None
    for tau in np.unique(np.quantile(P, np.linspace(0, 1, 401))):
        c = adaptive_pick(P, tau); rec = R[np.arange(len(c)), c].mean(); tok = TOK[np.arange(len(c)), c].mean()
        if rec >= target and (best is None or tok < best[2]):
            best = (float(tau), float(rec), float(tok))
    return best


def thresh_sets(data, t):
    out = []
    for x in data:
        keep = np.where(x["s"] >= t)[0]
        keep = keep if len(keep) else np.array([int(np.argmax(x["s"]))])
        out.append(keep[np.argsort(-x["s"][keep])])
    return out


def outcome(data, sets):
    rec = np.array([x["gold"][s].sum() == x["gold"].sum() for x, s in zip(data, sets)], float)
    tok = np.array([x["base"] + x["ptok"][s].sum() for x, s in zip(data, sets)], float)
    return rec, tok


def tune_thresh(data, target):
    need = np.array([x["s"][x["gold"]].min() for x in data])
    best = None
    for t in np.unique(need)[::-1]:
        rec, tok = outcome(data, thresh_sets(data, t))
        if rec.mean() >= target:
            best = (float(t), float(rec.mean()), float(tok.mean()))
            break  # highest threshold reaching the target = fewest tokens (sets are nested)
    return best


def boot(d, rng, n=2000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    global KS, TOPN
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=10, help="10 = pre-registered; larger = exploratory")
    args = ap.parse_args()
    KS = list(range(2, args.kmax + 1)); TOPN = max(10, min(20, args.kmax + 1))
    suf = "" if args.kmax == 10 else f"_kmax{args.kmax}"
    pool, dev, test = (load_split(s) for s in ("pool", "dev", "test"))
    idx = np.random.default_rng(0).permutation(len(pool)); cut = int(0.8 * len(pool))
    tr, va = [pool[i] for i in idx[:cut]], [pool[i] for i in idx[cut:]]
    M = {n: matrices(d) for n, d in (("tr", tr), ("va", va), ("dev", dev), ("test", test))}
    Atr, Ltr, Vtr, Ytr = M["tr"]; Ava, Lva, Vva, Yva = M["va"]
    sc_a = StandardScaler().fit(Atr); lr_a = LogisticRegression(max_iter=3000).fit(sc_a.transform(Atr), Ytr)
    TXtr = np.hstack([Atr, Ltr]); sc_t = StandardScaler().fit(TXtr)
    lr_t = LogisticRegression(max_iter=3000).fit(sc_t.transform(TXtr), Ytr)
    FXtr = np.hstack([Atr, Ltr, Vtr]); sc_f = StandardScaler().fit(FXtr)
    mlp, info = fit_mlp(sc_f.transform(FXtr).astype(np.float32), Ytr,
                        sc_f.transform(np.hstack([Ava, Lva, Vva])).astype(np.float32), Yva)

    def probs(model, split):
        A, L, V, _ = M[split]
        if model == "a1only":
            p = lr_a.predict_proba(sc_a.transform(A))[:, 1]
        elif model == "text":
            p = lr_t.predict_proba(sc_t.transform(np.hstack([A, L])))[:, 1]
        else:
            with torch.no_grad():
                p = torch.sigmoid(mlp(torch.tensor(sc_f.transform(np.hstack([A, L, V])).astype(np.float32)))).numpy()
        return p.reshape(-1, len(KS))

    models = ("a1only", "text", "full")
    tables = {"dev": topk_tables(dev), "test": topk_tables(test)}
    P = {(m, s): probs(m, s) for m in models for s in ("dev", "test")}
    rep = {"kmax": KS[-1], "pool_train": len(tr), "pool_val": len(va), "mlp": info,
           "a1_topk_full_recall": {s: {int(k): float(np.mean([x["gold"][np.argsort(-x["s"])[:k]].sum() == x["gold"].sum()
                                                               for x in d])) for k in range(1, 21)}
                                   for s, d in (("dev", dev), ("test", test))},
           "hops_test": {int(h): int(sum(x["hops"] == h for x in test)) for h in (2, 3, 4)}}
    rng = np.random.default_rng(1)
    sels, pol = {}, {}
    for target in TARGETS:
        row = {}
        t = tune_thresh(dev, target)
        a1_sets = thresh_sets(test, t[0])
        a1_rec, a1_tok = outcome(test, a1_sets)
        row["a1_thresh"] = {"dev": t, "test_recall": float(a1_rec.mean()), "test_tokens": float(a1_tok.mean()),
                            "test_mean_size": float(np.mean([len(s) for s in a1_sets]))}
        sels[f"a1_thresh_{target:g}"] = {x["id"]: [int(i) for i in s] for x, s in zip(test, a1_sets)}
        R, TOK = tables["test"]; per = {"a1_thresh": (a1_rec, a1_tok)}
        for m in models:
            b = tune_adaptive(P[(m, "dev")], *tables["dev"], target)
            if b is None:  # target unreachable within k <= max(KS) on dev: policy not testable
                row[f"adaptive_{m}"] = {"dev": None, "unreachable": True}
                continue
            c = adaptive_pick(P[(m, "test")], b[0])
            rec, tok = R[np.arange(len(c)), c].astype(float), TOK[np.arange(len(c)), c]
            per[m] = (rec, tok)
            row[f"adaptive_{m}"] = {"dev": b, "test_recall": float(rec.mean()), "test_tokens": float(tok.mean()),
                                    "test_mean_size": float(np.mean(np.array(KS)[c]))}
            sels[f"adaptive_{m}_{target:g}"] = {x["id"]: [int(i) for i in np.argsort(-x["s"])[:KS[ci]]]
                                                for x, ci in zip(test, c)}
        # 1. primary: adaptive text/full vs A1 threshold; 2. secondary: text/full vs a1only
        for m, ref in (("text", "a1_thresh"), ("full", "a1_thresh"), ("a1only", "a1_thresh"),
                       ("text", "a1only"), ("full", "a1only")):
            if m not in per or ref not in per:
                row[f"{m}_vs_{ref}"] = {"success": False, "not_testable": True}
                continue
            dt = per[m][1] - per[ref][1]; dr = per[m][0] - per[ref][0]
            ct, cr = boot(dt, rng), boot(dr, rng)
            row[f"{m}_vs_{ref}"] = {"tokens_diff": float(dt.mean()), "tokens_ci95": ct, "recall_diff": float(dr.mean()),
                                    "recall_ci95": cr, "success": bool(ct[1] < 0 and cr[1] >= 0)}
        # by hops (descriptive)
        row["by_hops"] = {int(h): {k: {"recall": float(v[0][[x["hops"] == h for x in test]].mean()),
                                       "tokens": float(v[1][[x["hops"] == h for x in test]].mean())}
                                   for k, v in per.items()} for h in (2, 3, 4)}
        # equal recall on test (each method's best threshold on test itself)
        row["frontier_test"] = {"a1_thresh": tune_thresh(test, target),
                                **{f"adaptive_{m}": tune_adaptive(P[(m, "test")], R, TOK, target) for m in models}}
        pol[f"recall_{target:g}"] = row
    rep["policies"] = pol
    rep["primary_success"] = any(pol[f"recall_{t:g}"][f"{m}_vs_a1_thresh"]["success"] for t in TARGETS for m in ("text", "full"))
    rep["secondary_success"] = any(pol[f"recall_{t:g}"][f"{m}_vs_a1only"]["success"] for t in TARGETS for m in ("text", "full"))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"results{suf}.json").write_text(json.dumps(rep, indent=1))
    (HERE / "results" / f"test_selections{suf}.json").write_text(json.dumps(sels))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
