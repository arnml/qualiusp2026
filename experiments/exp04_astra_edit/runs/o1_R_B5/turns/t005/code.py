# Hypothesis: the min-score ("hard") gold is usually the low-overlap bridge paragraph, and distractors beating it have high question overlap.
if "out1" not in globals(): print("pass 1 did not run"); raise SystemExit
S0 = np.asarray(out1["scores_original"], float); F["s0"] = S0
C = np.asarray(out1["cls_original"]); M = np.asarray(out1["mlp_original"]); print("C", C.shape, "M", M.shape)
def metrics(Fs, s, recall=0.90):
    Fs = Fs.assign(s=np.asarray(s)); g = Fs.groupby("qi")
    b3 = g.apply(lambda d: d.nlargest(3, "s").gold.sum() == d.gold.sum())
    mingold = g.apply(lambda d: d.loc[d.gold == 1, "s"].min())
    t = np.quantile(mingold.values, 1 - recall, method="lower")
    tokens = g.apply(lambda d: (d[d.s >= t] if (d.s >= t).any() else d.nlargest(1, "s")).ntok.sum())
    return dict(b3=b3.mean(), tokens=tokens.mean()), b3, tokens
for sp in ["A", "B", "all"]:
    sub = F if sp == "all" else F[F.split == sp]; print(sp, {k: round(v, 3) for k, v in metrics(sub, sub.s0)[0].items()})
F["rank0"] = F.groupby("qi").s0.rank(ascending=False)
F["mingold"] = F.qi.map(F[F.gold == 1].groupby("qi").s0.min())
G = F[F.gold == 1].assign(is_min=lambda d: d.s0 == d.mingold)
print(G.groupby(["type", "is_min"])[["overlap", "title_in_q", "has_ans", "rank0", "s0"]].mean().round(2))
D = F[F.gold == 0]; beat = D[D.s0 > D.mingold]
print("distractors beating hard gold: %d/%d; overlap %.2f vs rest %.2f; title_in_q %.2f vs %.2f" % (len(beat), len(D),
      beat.overlap.mean(), D[D.s0 <= D.mingold].overlap.mean(), beat.title_in_q.mean(), D[D.s0 <= D.mingold].title_in_q.mean()))
qi_arr = F.qi.values; A = (F.split == "A").values; gold = F.gold.values.astype(bool)
def cq(X):
    X = np.array(X, float, copy=True)
    for q in np.unique(qi_arr): X[qi_arr == q] -= X[qi_arr == q].mean(0)
    return X
s0c = cq(S0); 
for f in ["overlap", "title_in_q", "has_ans"]: print("within-q corr(score,%s)=%.3f" % (f, np.corrcoef(s0c, cq(F[f].values))[0, 1]))
ovc = cq(F.overlap.values)
for L in range(C.shape[1]):
    X = cq(C[:, L]); d = X[A][gold[A]].mean(0) - X[A][~gold[A]].mean(0); dn = d / np.linalg.norm(d)
    p = X[~A] @ dn; sepB = (p[gold[~A]].mean() - p[~gold[~A]].mean()) / p.std()
    print("L%d |d|=%.2f  B-sep=%.2f  corr(proj,score)=%.2f corr(proj,overlap)=%.2f" % (L, np.linalg.norm(d), sepB, np.corrcoef(X @ dn, s0c)[0, 1], np.corrcoef(X @ dn, ovc)[0, 1]))
