# Hypothesis: (rebuild) L4 MLP neurons that fire more for distractors than for hard (bridge) golds and write onto the relevance axis are the "bridge suppressors".
import numpy as np, pandas as pd, re, torch
F = pd.read_pickle(WORK / "feats.pkl"); S0 = np.load(WORK / "scores_orig.npy").astype(float)
C = np.load(WORK / "cls_orig.npy"); M = np.load(WORK / "mlp_orig.npy"); S2 = np.load(WORK / "scores_L5h0h6x2.npy").astype(float)
items = list(zip(F.qi.values.tolist(), F.pi.values.tolist()))
qi_arr = F.qi.values; gold = F.gold.values.astype(bool); dist = ~gold; ntok = F.ntok.values.astype(float); A = (qi_arr % 2 == 0)
QIDX = {q: np.where(qi_arr == q)[0] for q in np.unique(qi_arr)}; qA = [q for q in QIDX if q % 2 == 0]; qB = [q for q in QIDX if q % 2 == 1]
def cq(X):
    X = np.array(X, float, copy=True)
    for ix in QIDX.values(): X[ix] -= X[ix].mean(0)
    return X
mingold = np.zeros(len(F)); 
for ix in QIDX.values(): mingold[ix] = S0[ix][gold[ix]].min()
hardmask = gold & (S0 == mingold); beatmask = dist & (S0 > mingold)
def ev(s, qs, rcs=(0.85, 0.90, 0.95)):
    mins = np.array([s[QIDX[q]][gold[QIDX[q]]].min() for q in qs]); out = {}
    for rc in rcs:
        t = np.quantile(mins, 1 - rc, method="lower"); tk = []
        for q in qs:
            ix = QIDX[q]; k = s[ix] >= t; tk.append(ntok[ix][k].sum() if k.any() else ntok[ix][np.argmax(s[ix])])
        out["tok%d" % round(rc * 100)] = np.mean(tk)
    out["b3"] = np.mean([gold[QIDX[q][np.argsort(-s[QIDX[q]])[:3]]].sum() == gold[QIDX[q]].sum() for q in qs])
    m = np.isin(qi_arr, qs); hs, ds = s[m & hardmask], s[m & dist]; out["AUC"] = (hs[:, None] > ds[None, :]).mean()
    return {k: round(v, 3) for k, v in out.items()}
rng = np.random.default_rng(0)
def boot(Snew, qs, n=300, rc=0.90):
    d = []
    for _ in range(n):
        pick = rng.choice(qs, len(qs)); 
        # resampled questions: evaluate via concatenated per-question lists
        mins_n = np.array([Snew[QIDX[q]][gold[QIDX[q]]].min() for q in pick]); mins_o = np.array([S0[QIDX[q]][gold[QIDX[q]]].min() for q in pick])
        tn, to = np.quantile(mins_n, 1 - rc, method="lower"), np.quantile(mins_o, 1 - rc, method="lower")
        kn = np.mean([ntok[QIDX[q]][Snew[QIDX[q]] >= tn].sum() or ntok[QIDX[q]][np.argmax(Snew[QIDX[q]])] for q in pick])
        ko = np.mean([ntok[QIDX[q]][S0[QIDX[q]] >= to].sum() or ntok[QIDX[q]][np.argmax(S0[QIDX[q]])] for q in pick])
        d.append(kn - ko)
    return round(np.mean(d), 1), round(np.std(d), 1)
for sp, qs in [("A", qA), ("B", qB)]: print(sp, "orig", ev(S0, qs), "| h0h6x2", ev(S2, qs), "| tok90 Δ", boot(S2, qs))
# relevance axes per layer & per-neuron attribution (weights x activation differences; statistics only)
def axis(L):
    X = cq(C[:, L]); d = X[A & gold].mean(0) - X[A & dist].mean(0); return d / np.linalg.norm(d)
tab = {}
for L in [3, 4, 5]:
    Wo = load_original(f"bert.encoder.layer.{L}.output.dense.weight").numpy(); gam = load_original(f"bert.encoder.layer.{L}.output.LayerNorm.weight").numpy()
    w = (Wo * gam[:, None]).T @ axis(L + 1)
    hdA = (M[A & hardmask, L].mean(0) - M[A & dist, L].mean(0)) * w; hdB = (M[~A & hardmask, L].mean(0) - M[~A & dist, L].mean(0)) * w
    egA = (M[A & gold & ~hardmask, L].mean(0) - M[A & dist, L].mean(0)) * w
    tab[L] = pd.DataFrame(dict(hdA=hdA, hdB=hdB, egA=egA, w=w))
    neg = np.argsort(hdA)
    print("L%d hard-vs-distr contrib total A %.2f B %.2f | corr(hdA,hdB)=%.2f | top-100 neg: A %.2f B %.2f easygold %.2f | top-200 neg: A %.2f B %.2f"
          % (L, hdA.sum(), hdB.sum(), np.corrcoef(hdA, hdB)[0, 1], hdA[neg[:100]].sum(), hdB[neg[:100]].sum(), egA[neg[:100]].sum(), hdA[neg[:200]].sum(), hdB[neg[:200]].sum()))
pd.to_pickle(tab, WORK / "neuron_tab.pkl")
