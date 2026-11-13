The noise model sets the budget. With within-question noise of 0.25 (pass 3's level), the expected change is −0.31 pp and P(< −1 pp) ≈ 3%. At 0.55 (pass 4's level) it is −0.55 pp, with P(< −1 pp) ≈ 20% on a 200-question set. I want the final configuration at or below about 0.4.

The evidence says early injections are expensive: perturbations entering at layer 1 grow 2.5–4× per layer. So the final pass keeps pass 4's cheap-perturbation extras and drops the early ones.

**Pass 5 = pass 3 plus:**
- Layer-5 neurons by RMS up to 1,000.
- Layer-5 head 8.
- L4h9.
- Not the layer-1 heads 6 and 7, and not the extra layer-0/1 near-silent neurons.

**Decision rule:** adopt pass 5 if within-question score std ≤ ~0.4 and there is no net loss; otherwise revert to pass 3, which is tested clean.

```python
# Hypothesis: dropping P4's early-layer injections (L1h6, L1h7, extra L0-L3 near-silent neurons) while keeping L5 RMS-1000 + L5h8 + L4h9 brings within-q score noise to <=~0.4 with no net bg3 loss.
reset_all()
remove_neurons(5,L5_rms1000)
for h in [11,5,8]: remove_head(5,h)
for l,js in silent.items(): remove_neurons(l,js)      # the 140 fully CLS-silent neurons (as in P3)
remove_head(4,9)
print(cost())
R5=forward(items=items_order, model="edited", cls=True)
pickle.dump(R5, open(WORK/"pass5.pkl","wb"))
se5=np.asarray(R5['scores_edited']); report(se5)
d5=se5-so; print("score diff mean %+.2f |diff| %.2f within-q std %.3f"%(d5.mean(), np.abs(d5).mean(), qcenter(d5[:,None]).std()))
h5=bg3(se5,range(200)); print("P5 flips:", [(q,int(ho[q]),int(h5[q]),q in A) for q in range(200) if ho[q]!=h5[q]])
Ce5=np.asarray(R5['cls_edited'],np.float32); print("CLS cos per state P5:", [round(float(cos(Ce5[:,l],C[:,l]).mean()),5) for l in range(7)])
```