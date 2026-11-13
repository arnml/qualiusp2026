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
