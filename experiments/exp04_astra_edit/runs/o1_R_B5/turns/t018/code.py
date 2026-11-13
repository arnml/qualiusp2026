# Hypothesis: if body heads help, hard-gold ranks improve on B with a bootstrap-significant token reduction; ΔCLS shows what the heads write.
if "out2" not in globals(): raise SystemExit("pass 2 missing")
rng = np.random.default_rng(0)
qB = np.unique(qi_arr[~A]); qA = np.unique(qi_arr[A])
def boot(Snew, qs, n=400, rc=0.90):
    # paired bootstrap over questions of tokens@rc (edited - original)
    idx_by_q = {q: np.where(qi_arr == q)[0] for q in qs}; diffs = []
    for _ in range(n):
        pick = rng.choice(qs, len(qs)); idx = np.concatenate([idx_by_q[q] for q in pick])
        Fs = F.iloc[idx].copy(); Fs["qi"] = np.repeat(np.arange(len(pick)), [len(idx_by_q[q]) for q in pick])
        diffs.append(metrics(Fs, Snew[idx], rc)[0]["tokens"] - metrics(Fs, S0[idx], rc)[0]["tokens"])
    return np.mean(diffs), np.std(diffs)
for sp, qs in [("A", qA), ("B", qB)]:
    print("%s tok90 Δ = %.1f ± %.1f | tok85 Δ = %.1f ± %.1f" % ((sp,) + boot(S2, qs) + boot(S2, qs, rc=0.85)))
FBx = F[~A].assign(s2=S2[~A]); FBx["r2"] = FBx.groupby("qi").s2.rank(ascending=False)
wr = FBx[FBx.gold == 1].groupby("qi")[["rank0", "r2"]].max()
print("B worst-gold rank %.2f -> %.2f (improved %d, worse %d)" % (wr.rank0.mean(), wr.r2.mean(), (wr.r2 < wr.rank0).sum(), (wr.r2 > wr.rank0).sum()))
for t in ["bridge", "comparison"]:
    m = (F.type == t).values
    print(t, "b3 orig %.3f -> %.3f (all q)" % (metrics(F[m], S0[m])[0]["b3"], metrics(F[m], S2[m])[0]["b3"]))
dC = cq(C2[:, 6] - C[:, 6]); u, s_, vt = np.linalg.svd(dC, full_matrices=False)
print("ΔCLS6 within-q SVD energy top3:", np.round(s_[:3]**2 / (s_**2).sum(), 2), "| cos(top dir, r)=%.2f" % abs(vt[0] @ r_))
