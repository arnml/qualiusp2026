# Hypothesis: L4/L5 "title-match" neurons write onto the relevance axis with consistent sign; re-weighting them shifts hard-vs-beating by a non-trivial share of the gap.
tiy = cq(F.title_in_q.values.astype(float)); ra = partial(tiy, A)
sel = {}
for L in [3, 4, 5]:
    Ml = cq(M[:, L]); pc = np.nan_to_num(np.array([np.corrcoef(Ml[A, n], ra)[0, 1] for n in range(1536)]))
    w = tab[L].w.values; sgn = np.sign(pc * w)                       # +1: neuron pushes score up when title matches
    act = M[:, L]
    hbA = paired_gap(act, hardmask, beatmask, qA).mean(0) * w; hbB = paired_gap(act, hardmask, beatmask, qB).mean(0) * w
    hdA = paired_gap(act, hardmask, dist, qA).mean(0) * w; hdB = paired_gap(act, hardmask, dist, qB).mean(0) * w
    tot_hbA = hbA.sum(); tot_hbB = hbB.sum()
    for k in [60, 120, 200]:
        top = np.argsort(-np.abs(pc))[:k]
        print("L%d k=%3d | signed hard-beating gain per +100%% re-weight: A %+.3f B %+.3f (layer MLP total A %+.2f B %+.2f) | signed hard-all: A %+.3f B %+.3f | frac sgn>0 %.2f"
              % (L, k, (sgn[top] * hbA[top]).sum(), (sgn[top] * hbB[top]).sum(), tot_hbA, tot_hbB, (sgn[top] * hdA[top]).sum(), (sgn[top] * hdB[top]).sum(), (sgn[top] > 0).mean()))
    sel[L] = dict(pc=pc, sgn=sgn)
pd.to_pickle(sel, WORK / "title_neurons.pkl")
