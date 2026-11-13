# Hypothesis: final [CLS] carries a paragraph-length direction distinct from the readout; it's the handle for a length-normalisation edit.
X6c = cq(C[:, 6])
def cov_dir(X, f, use):  # covariance direction (statistic) of within-q centred X with feature f, on rows `use`
    d = (X[use] * (f[use] - f[use].mean())[:, None]).mean(0); return d / np.linalg.norm(d)
lenv = cov_dir(X6c, lnt, A); len_o = orth(lenv, [r_])
tiv = cov_dir(X6c, tic, A & dist); ti_o = orth(tiv, [r_])
print("cos(len, r)=%.2f cos(title, r)=%.2f cos(len_o, ti_o)=%.2f" % (lenv @ r_, tiv @ r_, len_o @ ti_o))
for nm, v in [("len_o", len_o), ("title_o", ti_o)]:
    p = X6c @ v
    print("%-8s B: corr(log ntok)=%.2f corr(title)=%.2f corr(score)=%.2f corr(ov)=%.2f | B gold-sep=%.2f sd | within-q sd=%.3f (score-axis sd %.3f)"
          % (nm, np.corrcoef(p[~A], lnt[~A])[0,1], np.corrcoef(p[~A], tic[~A])[0,1], np.corrcoef(p[~A], s0c[~A])[0,1], np.corrcoef(p[~A], ovc[~A])[0,1],
             (p[~A & gold].mean() - p[~A & dist].mean()) / p[~A].std(), p.std(), (X6c @ r_).std()))
# statistic-only scan (A): adding -len_o or +title_o projection to the recorded score
rows = []
for nm, v, sg in [("-len_o", len_o, -1), ("+title_o", ti_o, 1)]:
    p = C[:, 6] @ v; zz = (p - p.mean()) / cq(p).std()
    for beta in [0.25, 0.5, 1.0]:
        rows.append((nm, beta) + tuple(robust(FA, S0[A] + sg * beta * sdS * zz[A]).values()))
print(pd.DataFrame(rows, columns=["dir", "beta"] + list(robust(FA, S0[A]).keys())).to_string(index=False))
np.savez(WORK / "dirs_len_title.npz", len_o=len_o, ti_o=ti_o, r=r_)
