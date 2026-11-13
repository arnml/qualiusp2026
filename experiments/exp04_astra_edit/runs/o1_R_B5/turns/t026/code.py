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
