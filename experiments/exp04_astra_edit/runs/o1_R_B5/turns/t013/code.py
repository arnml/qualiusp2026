# Hypothesis: L4 anti-hard neurons are body-overlap detectors (not answer/title), while pro-hard L4 neurons track title-match/answer presence.
cA4, cB4, cg4, w4 = contribs[4]
M4c = cq(M[:, 4]); sd4 = M4c.std(0) + 1e-6
fc = {f: cq(F[f].values.astype(float)) for f in ["body_ov", "title_in_q", "has_ans", "ntok"]}
def fcorr(n): return {f: round(np.corrcoef(M4c[:, n], v)[0, 1], 2) for f, v in fc.items()}
neg = np.argsort(cA4)[:40]; pos = np.argsort(-cA4)[:40]
print("pro-hard total A=%.2f (top40 %.2f) B top40 %.2f | anti-hard top40 A %.2f B %.2f" % (cA4[cA4 > 0].sum(), cA4[pos].sum(), cB4[pos].sum(), cA4[neg].sum(), cB4[neg].sum()))
print("mean feature corr, anti-hard top40:", {f: round(np.mean([np.corrcoef(M4c[:, n], v)[0,1] for n in neg]), 2) for f, v in fc.items()})
print("mean feature corr, pro-hard  top40:", {f: round(np.mean([np.corrcoef(M4c[:, n], v)[0,1] for n in pos]), 2) for f, v in fc.items()})
print("anti-hard examples:", [(int(n), round(cA4[n], 3), round(cg4[n], 3), fcorr(n)) for n in neg[:4]])
print("pro-hard examples: ", [(int(n), round(cA4[n], 3), round(cg4[n], 3), fcorr(n)) for n in pos[:4]])
# specificity: anti-hard contribution per unit of general gold contribution
spec = cA4 - 0.25 * np.clip(cg4, 0, None)
print("neurons with cA<0 and cg<=0 (pure distractor-promoters):", int(((cA4 < 0) & (cg4 <= 0)).sum()), "sum cA %.3f" % cA4[(cA4 < 0) & (cg4 <= 0)].sum())
np.savez(WORK / "l4_neuron_stats.npz", cA=cA4, cB=cB4, cg=cg4, w=w4)
