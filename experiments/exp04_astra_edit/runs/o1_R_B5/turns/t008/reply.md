The final [CLS] is almost entirely the score direction: the gold mean-difference is 0.96 cosine with the readout. The neurons that separate gold from distractors are the same ones that drive the score, so there is no strong "ignored" gold direction in the plain mean contrast. The useful contrast must be hard-gold vs the distractors that beat it, which the in-flight block measures. In parallel, I'll measure an answer-presence direction and how all the candidate directions relate geometrically.

```python
# Hypothesis: an "answer present" direction exists at late-layer [CLS], partly orthogonal to the readout and to the overlap direction; it should lift hard golds.
hasans = F.has_ans.values.astype(bool); bridgeq = (F.type == "bridge").values
def pair_dir(X, use, pos, neg):  # within-question paired diff of means, averaged over questions having both
    diffs = []
    for q in np.unique(qi_arr[use]):
        m = qi_arr == q
        if (m & pos).any() and (m & neg).any(): diffs.append(X[m & pos].mean(0) - X[m & neg].mean(0))
    d = np.mean(diffs, 0); return d / np.linalg.norm(d), len(diffs)
X6raw = C[:, 6]
a6, na = pair_dir(X6raw, A & bridgeq, hasans, ~hasans & dist)          # answer-bearing vs non-answer distractors (bridge q, split A)
u6 = (cq(X6raw)[A & dist] * (ovc[A & dist] - ovc[A & dist].mean())[:, None]).mean(0); u6 /= np.linalg.norm(u6)
h6 = np.load(WORK / "hard_dir_L6.npy") if (WORK / "hard_dir_L6.npy").exists() else None
dirs = {"r": r, "ov": u6, "ans": a6} | ({"hard": h6} if h6 is not None else {})
print("n A questions for ans-dir:", na)
print(pd.DataFrame([[dirs[a] @ dirs[b] for b in dirs] for a in dirs], index=dirs, columns=dirs).round(2))
def orth(v, basis):
    for b in basis: v = v - (v @ b) * b
    return v / np.linalg.norm(v)
r_ = r / np.linalg.norm(r); ov_o = orth(u6, [r_])
a_o = orth(a6, [r_, ov_o])
for nm, v in [("ans", a6), ("ans _|_ r,ov", a_o)]:
    p = X6raw @ v
    pb = [p[(qi_arr == q) & hardmask].mean() - p[(qi_arr == q) & beatmask].mean() for q in np.unique(qi_arr[~A]) if ((qi_arr == q) & beatmask).any()]
    pall = [p[(qi_arr == q) & hardmask].mean() - p[(qi_arr == q) & dist].mean() for q in np.unique(qi_arr[~A])]
    print("%-14s B: hard-vs-beating paired mean/sd=%.2f frac>0=%.2f | hard-vs-all-distr frac>0=%.2f | within-q sd=%.3f corr(score)=%.2f corr(ov)=%.2f"
          % (nm, np.mean(pb)/np.std(pb), np.mean(np.array(pb) > 0), np.mean(np.array(pall) > 0), cq(p).std(), np.corrcoef(cq(p), s0c)[0,1], np.corrcoef(cq(p), ovc)[0,1]))
np.save(WORK / "dirs_L6.npy", np.stack([r_, u6, a6, a_o]))
```