Pass 1 worked: 4 passes remain, and inputs are truncated to 256 tokens. On average, gold paragraphs have higher title-in-question and answer-presence than distractors, and slightly higher overlap, so raw overlap is not anti-gold. While the turn-4 analysis output is in flight, I'll compute weight-side geometry and per-neuron statistics, which I'll need whatever that output shows.

```python
# Hypothesis: the final-layer gold signal is partly orthogonal to the pooler/classifier readout; some L4/L5 MLP neurons carry gold info the score ignores.
Wp = load_original("bert.pooler.dense.weight").numpy(); wc = load_original("classifier.weight").numpy().ravel()
r = Wp.T @ wc; r /= np.linalg.norm(r)                                   # linear readout direction (weights only)
X6 = cq(C[:, 6]); d6 = X6[A][gold[A]].mean(0) - X6[A][~gold[A]].mean(0); d6n = d6 / np.linalg.norm(d6)
dperp = d6n - (d6n @ r) * r; dperp /= np.linalg.norm(dperp)
print("cos(r, d_gold_L6)=%.3f" % (r @ d6n))
for nm, v in [("r", r), ("d_gold", d6n), ("d_perp", dperp)]:
    p = X6 @ v; pB = p[~A]
    print("%-7s corr(score)=%.2f corr(overlap)=%.2f  B gold-sep=%.2f sd" % (nm, np.corrcoef(p, s0c)[0,1], np.corrcoef(p, ovc)[0,1], (pB[gold[~A]].mean()-pB[~gold[~A]].mean())/pB.std()))
# hard gold vs distractors that beat it, along d_perp (B only)
hardmask = (F.gold == 1).values & (F.s0 == F.mingold).values; beatmask = (F.gold == 0).values & (F.s0 > F.mingold).values
pB = X6 @ dperp
print("B: d_perp hard-gold mean %.3f vs beating-distractors %.3f vs other distractors %.3f (sd %.3f)" % (pB[hardmask & ~A].mean(), pB[beatmask & ~A].mean(), pB[(~gold) & ~beatmask & ~A].mean(), pB[~A].std()))
def ntable(L):
    Mc = cq(M[:, L]); sd = Mc.std(0) + 1e-6
    dA = (Mc[A][gold[A]].mean(0) - Mc[A][~gold[A]].mean(0)) / sd; dB = (Mc[~A][gold[~A]].mean(0) - Mc[~A][~gold[~A]].mean(0)) / sd
    Z = (Mc - Mc.mean(0)) / sd
    cs = Z.T @ ((s0c - s0c.mean()) / s0c.std()) / len(s0c); co = Z.T @ ((ovc - ovc.mean()) / ovc.std()) / len(ovc)
    return pd.DataFrame(dict(dA=dA, dB=dB, corr_s=cs, corr_ov=co, sd=sd, mean=M[:, L].mean(0)))
for L in [3, 4, 5]:
    T = ntable(L); print("L%d corr(dA,dB)=%.2f  |dA|>0.5: %d  top+:" % (L, np.corrcoef(T.dA, T.dB)[0,1], (T.dA.abs() > 0.5).sum()))
    print(T.reindex(T.dA.abs().sort_values(ascending=False).index).head(6).round(2).to_string())
```