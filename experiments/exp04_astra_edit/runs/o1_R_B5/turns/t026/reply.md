Pass 4 is also close to null. Removing the 87 coverage neurons gives:
- A: tok90 −14.9 ± 19.3, AUC 0.720 → 0.724.
- B: tok90 −5.5 ± 13.9, tok85 −14.2 ± 16.9, AUC 0.641 → 0.646.
- Both-gold@3 is unchanged.

The direction is consistently slightly positive, but it is far below the SE. One pass remains.

Before using it, I want to know if the edit is directionally right but weak. Three checks on the measured score change Δs = S(α=1) − S0:
- whether it consistently favours hard golds over beating distractors;
- what features it tracks;
- how large it is relative to the score spread.

To pick a strength for the final pass, I'll extrapolate the **measured** change (S0 + k·Δs) on split A only. This is an extrapolation of observed outputs, not a computation of the network, and the final pass then tests the real network at that strength with B as confirmation.

```python
# Hypothesis: the coverage-neuron edit moves scores in the right direction but too weakly; Δs favours hard golds over beating distractors consistently (A and B).
S2 = np.load(WORK / "scores_L5h0h6x2.npy").astype(float)
for nm, Sx in [("cov a=1", S4), ("L5 h0h6 x2", S2)]:
    d = Sx - S0; dc = cq(d); s0c = cq(S0)
    gapA = np.array([d[(qi_arr == q) & hardmask].mean() - d[(qi_arr == q) & beatmask].mean() for q in qA if ((qi_arr == q) & beatmask).any()])
    gapB = np.array([d[(qi_arr == q) & hardmask].mean() - d[(qi_arr == q) & beatmask].mean() for q in qB if ((qi_arr == q) & beatmask).any()])
    print("%s: within-q sd(Δs)=%.3f vs sd(score)=%.3f | corr(Δs,score)=%.2f | hard-minus-beating Δs: A %.3f (frac>0 %.2f) B %.3f (frac>0 %.2f)"
          % (nm, dc.std(), s0c.std(), np.corrcoef(dc, s0c)[0, 1], gapA.mean(), (gapA > 0).mean(), gapB.mean(), (gapB > 0).mean()))
    print("   within-q corr(Δs, feature):", {f: round(np.corrcoef(dc, cq(F[f].values.astype(float)))[0, 1], 2) for f in ["overlap", "title_in_q", "has_ans", "ntok"]})
# linear extrapolation of the MEASURED change (dose scan on split A only; B shown for reference, not used to choose)
d4 = S4 - S0
for kk in [1, 2, 3, 4, 6]:
    print("k=%d  A %s | B %s" % (kk, ev(S0 + kk * d4, qA), ev(S0 + kk * d4, qB)))
```