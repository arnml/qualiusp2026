As suspected, `hard_perp` is essentially the negative score (corr −0.91), so its proxy "gains" are artifacts of compressing the score. The proxy tokens@90 curve is also very noisy (945 → 885 → 800 → 930). With about 100 questions, the 90% threshold is set by about 10 of them.

Before choosing anything I need steadier criteria:
- tokens averaged over 85/90/95% recall;
- a pooled AUC of hard-gold vs all distractors across questions, which is what a global threshold actually depends on.

Then I'll go to the component level. For the last layers' MLP neurons at [CLS], I'll attribute the hard-gold vs beating-distractor gap as Δactivation × (output weight · layer-6 relevance axis). This is a weight-space projection, not a forward simulation. It finds neurons that push beating distractors above hard golds, and I'll check that they replicate on B.

```python
# Hypothesis: specific late MLP neurons at [CLS] (lexical-match detectors) push high-overlap distractors above answer-bearing bridge golds; they replicate across splits.
def robust(Fs, s):
    s = np.asarray(s); out = {}
    for rc in [0.85, 0.90, 0.95]: out["tok%d" % int(rc*100)] = metrics(Fs, s, rc)[0]["tokens"]
    out["b3"] = metrics(Fs, s)[0]["b3"]; hm = (Fs.gold == 1).values & (Fs.s0 == Fs.mingold).values
    hs, ds = s[hm], s[(Fs.gold == 0).values]
    out["AUC_hard_vs_distr_pooled"] = (hs[:, None] > ds[None, :]).mean()
    return {k: round(v, 3 if k in ("b3",) or k.startswith("AUC") else 1) for k, v in out.items()}
print("orig A", robust(FA, S0[A])); print("orig B", robust(FB, S0[~A]))
m6 = cq(C[:, 6])[A & gold].mean(0) - cq(C[:, 6])[A & dist].mean(0); m6 /= np.linalg.norm(m6)   # L6 relevance axis (≈ readout)
def paired(act, use, pos, neg):
    out = []
    for q in np.unique(qi_arr[use]):
        m = qi_arr == q
        if (m & pos).any() and (m & neg).any(): out.append(act[m & pos].mean(0) - act[m & neg].mean(0))
    return np.array(out)
contribs = {}
for L in [3, 4, 5]:
    mL = m6 if L == 5 else (lambda X: (X[A & gold].mean(0) - X[A & dist].mean(0)) / np.linalg.norm(X[A & gold].mean(0) - X[A & dist].mean(0)))(cq(C[:, L + 1]))
    Wo = load_original(f"bert.encoder.layer.{L}.output.dense.weight").numpy(); gam = load_original(f"bert.encoder.layer.{L}.output.LayerNorm.weight").numpy()
    w = (Wo * gam[:, None]).T @ mL                                         # per-neuron write onto relevance axis (weights only)
    dA = paired(M[:, L], A, hardmask, beatmask).mean(0); dB = paired(M[:, L], ~A, hardmask, beatmask).mean(0)
    gA = paired(M[:, L], A, gold, dist).mean(0)
    cA, cB, cg = dA * w, dB * w, gA * w                                     # contribution to (hard - beating) and (gold - distr) along axis
    contribs[L] = (cA, cB, cg, w)
    order = np.argsort(cA)[:8]
    print("L%d: total hard-beating contrib A=%.2f B=%.2f | corr(cA,cB)=%.2f | most negative (anti-hard) neurons:" % (L, cA.sum(), cB.sum(), np.corrcoef(cA, cB)[0,1]))
    print("   ", [(int(n), round(cA[n], 3), round(cB[n], 3), round(cg[n], 3)) for n in order])
    print("    sum of 30 most negative: A %.2f  B %.2f  gold-vs-distr(A) %.2f" % (cA[np.argsort(cA)[:30]].sum(), cB[np.argsort(cA)[:30]].sum(), cg[np.argsort(cA)[:30]].sum()))
```