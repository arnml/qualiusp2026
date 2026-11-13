Layer-4 head 9 stands out as a no-op candidate:
- Highest sink mass in its layer (0.45).
- Lowest OV norm of any head in layers 0–4 (2.2).
- Lowest direct readout in layer 4 (0.07).
- No positional, lexical or cross-segment role.

Its no-op score (0.20) is about twice the next ones (L1h6 0.11, L1h7 0.08), so the evidence for the others is weaker.

I'll budget the score perturbation. Pass 3's within-question score-difference std is 0.25. With 12 questions inside a ±1 margin, I want the total to stay below about 0.4.

**Pass 4 = pass 3 plus a moderate step**
- Layer 5: neurons by RMS up to 1,000 (perturbation ~11% of LN scale, vs. ~7% now).
- Layer 5: remove head 8 (lowest remaining readout-OV and low OV norm).
- Layers 0–3: also remove neurons with max |output| < 0.02 at [CLS] (141 more).
- Early heads: remove L4h9, L1h6, L1h7.

Localization: if the [CLS] at layer 4's output moves much more than in pass 3, the early heads are responsible.

```python
# Hypothesis: P3 + (L5 RMS-1000, L5 head 8, early max|a|<0.02 neurons, heads L4h9/L1h6/L1h7) keeps within-q score-diff std < ~0.4 and no net bg3 loss.
reset_all()
L5_rms1000=[int(j) for j in oR[:1000]]; remove_neurons(5,L5_rms1000)
for h in [11,5,8]: remove_head(5,h)
silent2={l:[int(j) for j in np.where(np.abs(M[:,l]).max(0)<0.02)[0]] for l in range(4)}
for l,js in silent2.items(): remove_neurons(l,js)
for (l,h) in [(4,9),(1,6),(1,7)]: remove_head(l,h)
print({l:len(v) for l,v in silent2.items()}, cost())
R4=forward(items=items_order, model="edited", cls=True, mlp=True)
pickle.dump(R4, open(WORK/"pass4.pkl","wb"))
se4=np.asarray(R4['scores_edited']); report(se4)
d4=se4-so; print("score diff mean %+.2f |diff| %.2f within-q std %.3f"%(d4.mean(), np.abs(d4).mean(), qcenter(d4[:,None]).std()))
Ce4=np.asarray(R4['cls_edited'],np.float32)
print("CLS cos per state P4:", [round(float(cos(Ce4[:,l],C[:,l]).mean()),5) for l in range(7)])
```