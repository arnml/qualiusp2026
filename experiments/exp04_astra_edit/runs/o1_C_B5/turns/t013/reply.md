The ranking from split A matches the all-data ranking (overlap 1.00), and the predicted direct effect is identical on split B (std 0.058 for 1,300 neurons). So the selection doesn't overfit A.

One caveat: low-importance neurons carry most of the MLP's constant mean output (1.80 of 1.98), so removing them adds a fixed offset before the final LayerNorm. Pass 2 will show whether that matters.

**Pass 2 configuration**
- Drop layer 1.
- Layer 5: remove the 1,300 lowest-importance neurons (ranked on A).
- Layer 5: remove heads 11, 5 and 8 (lowest direct readout-OV: 0.05, 0.08, 0.13).

I'm requesting [CLS] states and MLP activations so I can separate layer-1 damage (visible at layer 4's output, before any layer-5 edit) from layer-5 damage.

```python
# Hypothesis: layer 1 is skippable (its positional/lexical roles duplicated in layer 2) and layer 5 can lose 1300 neurons + 3 low-readout heads; pass 2 tests both, separable via per-layer CLS.
reset_all()
remove_layer(1)
L5_drop=[int(j) for j in oA[:1300]]; remove_neurons(5, L5_drop)
for h in [11,5,8]: remove_head(5,h)
print(cost())
R2=forward(items=items_order, model="edited", cls=True, mlp=True)
pickle.dump(R2, open(WORK/"pass2.pkl","wb"))
se=np.asarray(R2['scores_edited']); print({k:(v.shape if hasattr(v,'shape') else v) for k,v in R2.items()})
def report(se):
    for nm,qs in [('A',sorted(A)),('B',sorted(B))]:
        ho=bg3(so,qs); he=bg3(se,qs); 
        rho=np.mean([__import__('scipy').stats.spearmanr([so[idx[(qi,p)]] for p in range(len(practice[qi]['paragraphs']))],[se[idx[(qi,p)]] for p in range(len(practice[qi]['paragraphs']))])[0] for qi in qs if len(practice[qi]['paragraphs'])>2])
        print(f"{nm}: orig {ho.mean():.3f} edited {he.mean():.3f}  lost {((ho==1)&(he==0)).sum()} gained {((ho==0)&(he==1)).sum()}  mean within-q spearman {rho:.3f}")
report(se)
```