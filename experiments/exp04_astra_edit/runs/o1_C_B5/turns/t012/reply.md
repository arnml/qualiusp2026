At the [CLS] position, layers 0–3 contribute almost nothing to the readout direction (within-question std 0.01–0.07, against 1.2 for layer 4 and 2.8–3.1 for layer 5). That doesn't prove they're skippable, because they build token representations.

Layer 1 is the best whole-layer candidate:
- Smallest [CLS] update (|Δ| = 2.05, cos 0.98 with its input).
- Sparsest MLP at [CLS].
- Its positional and lexical heads have close counterparts in layer 2 (previous/next/same-token heads), so its function looks partly redundant.

**Plan for pass 2:** drop layer 1, and thin layer 5 using importance computed on **split A only**. The [CLS] outputs per layer will separate the two effects, since layer-1 damage shows up before layer 5. First I'll recompute the layer-5 ranking on A and check the mean-shift budget.

```python
# Hypothesis: layer-5 neuron ranking from split A alone matches the all-data ranking; removing ~1300 low-importance neurons shifts the CLS residual mean by a small fraction of the full MLP mean output.
def qcenter_mask(X,mask):
    Xc=X[mask].copy(); q=qid[mask]
    for qi in np.unique(q): m=q==qi; Xc[m]-=Xc[m].mean(0)
    return Xc
a5A=a5[inA]; impA=qcenter_mask(a5,inA).std(0)*np.abs(proj)
oA=np.argsort(impA)
full_mean_shift=np.linalg.norm((a5A.mean(0)*Wout5*g5[:,None]).sum(1))
for k in [1000,1200,1300,1400]:
    S=oA[:k]; sh=np.linalg.norm((a5A.mean(0)[S]*Wout5[:,S]*g5[:,None]).sum(1))
    dvA=qcenter_mask((a5[:,S]*proj[S]).sum(1)[:,None],inA).std(); dvB=qcenter_mask((a5[:,S]*proj[S]).sum(1)[:,None],~inA).std()
    print(k, "overlap with all-data set %.2f"%(len(set(S)&set(o5[:k]))/k), " within-q proj std A %.3f B %.3f"%(dvA,dvB), " mean-shift %.2f of full %.2f"%(sh,full_mean_shift))
```