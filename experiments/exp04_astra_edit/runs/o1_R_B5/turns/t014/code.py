# Hypothesis: the score rises with paragraph length among distractors (un-normalised lexical evidence); L5 heads 0/6/11 read the paragraph rather than the question.
nt = F.ntok.values.astype(float); lnt = cq(np.log(nt))
print("within-q corr(score, log ntok): distractors %.2f | golds %.2f | all %.2f" % (np.corrcoef(s0c[dist], lnt[dist])[0,1], np.corrcoef(s0c[gold], lnt[gold])[0,1], np.corrcoef(s0c, lnt)[0,1]))
print("within-q corr(overlap, log ntok) distractors %.2f; corr(score, log ntok | overlap-partialled, distr) %.2f" % (np.corrcoef(ovc[dist], lnt[dist])[0,1],
      np.corrcoef(s0c[dist] - np.polyval(np.polyfit(ovc[dist], s0c[dist], 1), ovc[dist]), lnt[dist])[0,1]))
print("gold vs distractor log-ntok gap (within-q, A/B): %.2f / %.2f" % (lnt[A & gold].mean() - lnt[A & dist].mean(), lnt[~A & gold].mean() - lnt[~A & dist].mean()))
# where is length at CLS? per layer corr of best single coordinate and of length-cov direction (A) tested on B
for L in range(1, 7):
    X = cq(C[:, L]); dl = (X[A] * lnt[A, None]).mean(0); dl /= np.linalg.norm(dl); p = X @ dl
    print("L%d length-dir: corr(log ntok) B=%.2f | corr(score) B=%.2f | cos(m6)=%s" % (L, np.corrcoef(p[~A], lnt[~A])[0,1], np.corrcoef(p[~A], s0c[~A])[0,1], ("%.2f" % (dl @ m6)) if L == 6 else "-"))
# what do L5 heads 0, 6, 11 and L4 heads attend to from CLS? (q0, 8 inputs) mass on CLS itself / [SEP]s / body
for L, hs in [(5, [0, 6, 11]), (4, [1, 4, 10, 11])]:
    for h in hs:
        cls_m = np.mean([att[k, L, h, 0, 0] for k in range(8)]); sep_m = np.mean([att[k, L, h, 0, sep_pos[k]].sum() for k in range(8)])
        print("L%d h%d: CLS->self %.2f  ->[SEP]s %.2f  ->question %.2f  ->title %.2f  ->body %.2f" % (L, h, cls_m, sep_m, res[L, h, 0], res[L, h, 1], 1 - cls_m - sep_m - res[L, h, 0] - res[L, h, 1]))
