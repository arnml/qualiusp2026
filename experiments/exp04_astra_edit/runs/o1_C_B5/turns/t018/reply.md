Two findings change the plan:
- **Ranking layer-5 neurons by within-question variance was a mistake.** The 1,300-neuron set mostly consisted of neurons sitting at the GELU floor (~−0.16). Together they removed a near-constant vector of norm 4.3 per item, about 44% of the scale of the residual entering the final LayerNorm.
- **Within-question Spearman is ~0.76 regardless of how intact layer 5's input was.** That implicates the layer-5 edit as a major source of ranking damage, on top of the layer-1 drop.

Ranking by RMS contribution instead gives far smaller perturbations: 700 neurons ≈ 6–7% of the scale.

**Pass 3 (conservative base)**
- Layer 5: remove the 700 lowest-RMS neurons (ranked on A).
- Layer 5: remove heads 11 and 5 (lowest direct readout-OV).
- Layers 0–3: remove the 140 neurons that are completely silent at [CLS] (max |output| < 0.005 on all items).

```python
# Hypothesis: a base edit with small, well-characterized perturbations (L5 RMS-700 neurons, L5 heads 11&5, 140 CLS-silent early neurons) keeps within-q ranking ~intact (spearman>0.97, no net bg3 loss).
reset_all()
L5_rms=[int(j) for j in oR[:700]]; remove_neurons(5,L5_rms)
for h in [11,5]: remove_head(5,h)
silent={l:[int(j) for j in np.where(np.abs(M[:,l]).max(0)<0.005)[0]] for l in range(4)}
for l,js in silent.items(): remove_neurons(l,js)
print({l:len(v) for l,v in silent.items()}, cost())
NOTES.write_text(NOTES.read_text()+"""
## Split: rng(0) permutation; A=perm[:100] selection, B=perm[100:] confirmation.
## Pass1: original diagnostics. bg3 A .91 B .98 all .945. Decision built in L4-L5 (CLS mean-diff dir bg3: L3out .25, L4out .50, final .98).
## Pass2: drop L1 + L5 1300 neurons (ranked by within-q var) + L5 heads 11,5,8 -> FAIL (A -7pp, B -17pp, spearman .76).
  - L1 drop damages token reps: L4-out CLS discriminability .70->.51 though CLS cos at L2 out .997 (CLS insensitive early).
  - L5 var-ranked neurons were GELU-floor constants: removed ~44% of LN-scale as constant offset -> damage. Use RMS ranking instead.
## Pass3 plan: L5 RMS-700 + L5 heads 11,5 + 140 CLS-silent early neurons (L0-3, max|a|<.005).
""")
R3=forward(items=items_order, model="edited", cls=True, mlp=True)
pickle.dump(R3, open(WORK/"pass3.pkl","wb"))
se3=np.asarray(R3['scores_edited']); report(se3)
print("score diff mean %+.2f |diff| %.2f"%((se3-so).mean(), np.abs(se3-so).mean()))
```