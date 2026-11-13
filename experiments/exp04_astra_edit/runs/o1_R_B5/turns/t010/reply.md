That result changes the picture. At layer 6, every raw difference-of-means direction (overlap, answer, hard-gold) is the same axis as the readout, with |cos| between 0.94 and 0.99. The final [CLS] is close to one-dimensional for relevance. The "hard_perp" direction is then likely just a residue of that axis (corr −0.98 with score), not new information. The answer direction orthogonal to readout and overlap carries nothing (paired frac>0 0.48).

So I'll stop chasing raw mean differences. Two better questions:
1. How much of the token cost comes from per-question score offsets (cross-question calibration under a global threshold) versus within-question ranking? The oracle centering in the code below is a diagnostic only, never a target to write into weights.
2. After removing the score axis, is any gold information left in layers 4–6?

```python
# Hypothesis: much of the token cost comes from question-level score offsets (global threshold), and/or gold info survives at L4/L5 once the score axis is partialled out.
qmean = F.groupby("qi").s0.transform("mean").values
dmean = F.assign(sd_=np.where(gold, np.nan, S0)).groupby("qi").sd_.transform("mean").values
print("var of per-q mean distractor score %.2f vs within-q distractor var %.2f" % (pd.Series(dmean).groupby(qi_arr).first().var(), s0c[dist].var()))
print("corr over q: hard-gold score vs mean distractor score = %.2f" % np.corrcoef(F[hardmask].groupby('qi').s0.first(), pd.Series(dmean[hardmask], index=qi_arr[hardmask]).groupby(level=0).first())[0,1])
for nm, s in [("orig", S0), ("oracle q-centred", S0 - qmean), ("oracle q-rank", -F.rank0.values)]:
    print("%-17s A %s | B %s" % (nm, {k: round(v,1 if k=='tokens' else 3) for k,v in metrics(FA, s[A])[0].items()}, {k: round(v,1 if k=='tokens' else 3) for k,v in metrics(FB, s[~A])[0].items()}))
# score-partialled gold direction per layer (cov statistics; direction from A, tested on B)
for L in [3, 4, 5, 6]:
    X = cq(C[:, L]); b = (X * s0c[:, None]).mean(0) / s0c.var(); Xr = X - np.outer(s0c, b)
    d = Xr[A & gold].mean(0) - Xr[A & dist].mean(0); dn = d / np.linalg.norm(d); p = Xr @ dn
    pb = [p[(qi_arr==q)&hardmask].mean() - p[(qi_arr==q)&beatmask].mean() for q in np.unique(qi_arr[~A]) if ((qi_arr==q)&beatmask).any()]
    sepB = (p[~A & gold].mean() - p[~A & dist].mean()) / p[~A].std()
    print("L%d resid-gold dir: B gold-sep=%.2f sd | B hard>beating frac=%.2f | corr ntok=%.2f has_ans=%.2f title=%.2f ov=%.2f"
          % (L, sepB, np.mean(np.array(pb) > 0), np.corrcoef(p, ntc)[0,1], np.corrcoef(p, hac)[0,1], np.corrcoef(p, tic)[0,1], np.corrcoef(p, ovc)[0,1]))
```