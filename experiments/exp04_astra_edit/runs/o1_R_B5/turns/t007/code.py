# Hypothesis: hard (answer-bearing bridge) golds differ from the distractors that outrank them along a direction the readout under-uses; overlap among distractors drives score noise.
dist = ~gold
# (1) overlap direction among distractors only (within-q centred): how much of distractor score is overlap-driven?
for L in [4, 5, 6]:
    X = cq(C[:, L]); XA = X[A & dist]; oA = ovc[A & dist]
    u = (XA * (oA - oA.mean())[:, None]).mean(0); u /= np.linalg.norm(u)          # covariance direction with overlap (stat, not a fit)
    pu = X @ u
    print("L%d overlap-dir: cos(r)=%s corr(proj,score|distr,B)=%.2f corr(proj,ov|distr,B)=%.2f  B gold-sep=%.2f" % (L,
          ("%.2f" % (r @ u)) if L == 6 else "  - ", np.corrcoef(pu[~A & dist], s0c[~A & dist])[0,1], np.corrcoef(pu[~A & dist], ovc[~A & dist])[0,1],
          (pu[~A][gold[~A]].mean() - pu[~A][dist[~A]].mean()) / pu[~A].std()))
print("corr(score, overlap | distractors) = %.2f" % np.corrcoef(s0c[dist], ovc[dist])[0,1])
# (2) hard-gold vs beating-distractor direction, paired within question (split A), tested on B
def hard_dir(L, use):
    X = C[:, L]; diffs = []
    for q in np.unique(qi_arr[use]):
        m = qi_arr == q; hb = m & beatmask
        if hb.any(): diffs.append(X[m & hardmask].mean(0) - X[hb].mean(0))
    return np.mean(diffs, 0), len(diffs)
for L in [3, 4, 5, 6]:
    h, n = hard_dir(L, A); hn = h / np.linalg.norm(h)
    X = C[:, L]; p = X @ hn
    # paired B test: per question, hard gold minus mean beating distractor along h
    pb = [p[(qi_arr == q) & hardmask].mean() - p[(qi_arr == q) & beatmask].mean() for q in np.unique(qi_arr[~A]) if ((qi_arr == q) & beatmask).any()]
    extra = ""
    if L == 6:
        hp = hn - (hn @ r) * r; hp /= np.linalg.norm(hp); pp = X @ hp
        pbp = [pp[(qi_arr == q) & hardmask].mean() - pp[(qi_arr == q) & beatmask].mean() for q in np.unique(qi_arr[~A]) if ((qi_arr == q) & beatmask).any()]
        extra = " | cos(r)=%.2f, orth-part B paired mean/sd=%.2f (frac>0 %.2f)" % (hn @ r, np.mean(pbp) / np.std(pbp), np.mean(np.array(pbp) > 0))
        np.save(WORK / "hard_dir_L6.npy", hn); np.save(WORK / "hard_dir_L6_perp.npy", hp)
    print("L%d hard-dir (nA=%d): B paired mean/sd=%.2f, frac>0=%.2f, corr(score)=%.2f, corr(ov)=%.2f%s" % (L, n, np.mean(pb) / np.std(pb), np.mean(np.array(pb) > 0),
          np.corrcoef(cq(p), s0c)[0,1], np.corrcoef(cq(p), ovc)[0,1], extra))
print(pd.DataFrame({"hard_gold": F[hardmask][["has_ans","overlap","title_in_q","ntok"]].mean(), "beating_distr": F[beatmask][["has_ans","overlap","title_in_q","ntok"]].mean()}).round(2).T)
