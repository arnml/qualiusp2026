# Hypothesis: the hard-gold/beating-distractor flip happens at a specific layer; some late neurons carry title-match / answer-presence beyond body-overlap & score.
def paired_gap(X, pos, neg, qs):
    return np.array([X[QIDX[q]][pos[QIDX[q]]].mean(0) - X[QIDX[q]][neg[QIDX[q]]].mean(0) for q in qs if neg[QIDX[q]].any() and pos[QIDX[q]].any()])
for L in range(1, 7):
    ax = axis(L); p = C[:, L] @ ax; sdp = cq(p).std()
    gb = paired_gap(p, hardmask, beatmask, qB) / sdp; gd = paired_gap(p, hardmask, dist, qB) / sdp
    print("C%d (B) hard-beating %.2f sd (frac>0 %.2f) | hard-all distr %.2f sd (frac>0 %.2f)" % (L, gb.mean(), (gb > 0).mean(), gd.mean(), (gd > 0).mean()))
# answer-presence / title-match carriers beyond body overlap & score (partial correlations; A selects, B replicates)
body_ov = []
STOP = set("the a an of in on at to for by with from and or is was are were be been which what who whom whose when where how did does do that this as it its his her their they he she than also other".split())
content = lambda s: set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1)
for q in practice:
    qw = content(q["question"])
    for p in q["paragraphs"]: body_ov.append(len(qw & content(p.split(". ", 1)[-1])) / max(1, len(qw)))
F["body_ov"] = body_ov
Z = np.column_stack([cq(S0), cq(F.body_ov.values), cq(np.log(ntok))])
def partial(y, use):  # residualise y on Z via column-wise projection (statistic, used only for correlation)
    Zu = Z[use]; b = np.linalg.lstsq(Zu, y[use], rcond=None)[0]; return y[use] - Zu @ b
for f in ["has_ans", "title_in_q"]:
    fy = cq(F[f].values.astype(float))
    for L in [3, 4, 5]:
        Ml = cq(M[:, L]); ra = partial(fy, A); rb = partial(fy, ~A)
        MA = np.column_stack([partial(Ml[:, n], A) for n in range(Ml.shape[1])]) if False else None
        cA = np.array([np.corrcoef(Ml[A, n], ra)[0, 1] for n in range(1536)]); cB = np.array([np.corrcoef(Ml[~A, n], rb)[0, 1] for n in range(1536)])
        cA, cB = np.nan_to_num(cA), np.nan_to_num(cB); top = np.argsort(-np.abs(cA))[:20]
        print("%s L%d: corr(cA,cB)=%.2f | top20 |cA| mean %.2f, same-sign B mean %.2f" % (f, L, np.corrcoef(cA, cB)[0, 1], np.abs(cA[top]).mean(), (np.sign(cA[top]) * cB[top]).mean()))
