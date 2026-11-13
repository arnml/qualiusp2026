
import numpy as np, pandas as pd, re, torch
F = pd.read_pickle(WORK / "feats.pkl"); S0 = np.load(WORK / "scores_orig.npy").astype(float)
qi_arr = F.qi.values; gold = F.gold.values.astype(bool); dist = ~gold; ntok = F.ntok.values.astype(float); A = (qi_arr % 2 == 0)
items = list(zip(F.qi.values.tolist(), F.pi.values.tolist()))
NQ = qi_arr.max() + 1; POS = F.pi.values
def pad(x, fill):
    P = np.full((NQ, 10), fill, dtype=float); P[qi_arr, POS] = x; return P
G = pad(gold.astype(float), 0) > 0; NT = pad(ntok, 0); VALID = pad(np.ones(len(F)), 0) > 0
qA = np.arange(0, NQ, 2); qB = np.arange(1, NQ, 2)
def mins_of(s): Sp = pad(s, -np.inf); return np.where(G, Sp, np.inf).min(1), Sp
H0 = np.zeros(len(F), bool)
m0, Sp0 = mins_of(S0); H0 = gold & (S0 == m0[qi_arr])                       # hard gold under ORIGINAL model
hardmask = H0; beatmask = dist & (S0 > m0[qi_arr])
def ev(s, qs, rcs=(0.85, 0.90, 0.95)):
    mins, Sp = mins_of(s); out = {}
    for rc in rcs:
        t = np.quantile(mins[qs], 1 - rc, method="lower"); k = Sp[qs] >= t
        tk = (NT[qs] * k).sum(1); none = ~k.any(1); tk[none] = NT[qs][none, np.argmax(Sp[qs][none], 1)]
        out["tok%d" % round(rc * 100)] = tk.mean()
    top3 = np.argsort(-Sp[qs], 1)[:, :3]; out["b3"] = (np.take_along_axis(G[qs], top3, 1).sum(1) == G[qs].sum(1)).mean()
    m = np.isin(qi_arr, qs); hs, ds = s[m & hardmask], s[m & dist]; out["AUC"] = (hs[:, None] > ds[None, :]).mean()
    return {k: round(float(v), 3) for k, v in out.items()}
def boot(Snew, qs, n=500, rc=0.90, seed=0):
    rng = np.random.default_rng(seed); pick = rng.choice(qs, (n, len(qs)))
    res = []
    for s in (Snew, S0):
        mins, Sp = mins_of(s); t = np.quantile(mins[pick], 1 - rc, axis=1, method="lower")
        k = Sp[pick] >= t[:, None, None]; tk = (NT[pick] * k).sum(2)
        none = ~k.any(2); am = np.argmax(Sp[pick], 2); tk = np.where(none, np.take_along_axis(NT[pick], am[..., None], 2)[..., 0], tk)
        res.append(tk.mean(1))
    d = res[0] - res[1]; return round(float(d.mean()), 1), round(float(d.std()), 1)
def cq(X):
    X = np.array(X, float, copy=True); Q = qi_arr
    for q in range(NQ): ix = Q == q; X[ix] -= X[ix].mean(0)
    return X
