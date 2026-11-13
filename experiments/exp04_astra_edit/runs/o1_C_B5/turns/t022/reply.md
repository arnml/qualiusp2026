Pass 4 is a mixed result:
- **Split A:** 2 lost, 1 gained.
- **Split B:** no changes.
- **Net:** −1 of 200 questions (−0.5 pp), with Spearman 0.967 to 0.974.
- **Perturbation:** within-question score-difference std is 0.55, over my 0.4 budget.

That's a meaningful risk of losing more than 1 pp on unseen questions, so I don't want to submit pass 4 as is.

The [CLS] perturbation grows already at the layer-1 output (1−cos goes from 1e-5 to 1.9e-4) and is ~4× larger at layer 4's output. That points at the early edits (L1 heads, extra early neurons, L4h9) at least as much as the extra layer-5 edits.

With one pass left, I'll localize before choosing the final step. I'll compare pass-4 vs pass-3 changes per layer and see whether the extra score change tracks upstream (pre-layer-5) perturbation.

```python
# Hypothesis: P4's extra score noise is driven mainly by the early edits (L1 heads, L4h9), visible as larger pre-L5 CLS perturbation; the extra L5 neuron/head removals contribute less.
Me4=np.asarray(R4['mlp_edited'],np.float32)
e=d4-d3
r5_3=1-cos(Ce3[:,5],C[:,5]); r5_4=1-cos(Ce4[:,5],C[:,5])
print("corr(|extra dscore|, L4-out perturbation P4) = %.2f"%np.corrcoef(np.abs(qcenter(e[:,None]).ravel()), r5_4)[0,1])
print("corr(|extra dscore|, increase in L4-out perturbation) = %.2f"%np.corrcoef(np.abs(qcenter(e[:,None]).ravel()), r5_4-r5_3)[0,1])
for l in range(5):
    print(f"L{l} MLP@CLS corr: P3 {np.corrcoef(Me3[:,l].ravel(),M[:,l].ravel())[0,1]:.5f}  P4 {np.corrcoef(Me4[:,l].ravel(),M[:,l].ravel())[0,1]:.5f}")
k5b=np.setdiff1d(np.arange(1536),L5_rms1000)
for nm,Mx in [('P3',Me3),('P4',Me4)]:
    dd=Mx[:,5][:,k5b]-M[:,5][:,k5b]; print(f"{nm}: L5 MLP input-driven change on P4-kept neurons rel {np.linalg.norm(dd)/np.linalg.norm(M[:,5][:,k5b]):.3f}")
# which questions flipped in P4
ho=bg3(so,range(200)); h4=bg3(se4,range(200)); h3=bg3(se3,range(200))
print("P4 flips (q, orig->P4, in A?):", [(q,int(ho[q]),int(h4[q]),q in A) for q in range(200) if ho[q]!=h4[q]])
print("P3 flips:", [(q,int(ho[q]),int(h3[q]),q in A) for q in range(200) if ho[q]!=h3[q]])
```