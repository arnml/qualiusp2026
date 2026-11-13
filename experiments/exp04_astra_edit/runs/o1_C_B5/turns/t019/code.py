# Hypothesis: early CLS-silent neuron removal leaves L0-L4 CLS ~identical (cos>0.999); remaining perturbation comes from L5 heads/neurons, which are small -> room to push L5 harder.
Ce3=np.asarray(R3['cls_edited'],np.float32); Me3=np.asarray(R3['mlp_edited'],np.float32)
for l in range(7):
    c=cos(Ce3[:,l],C[:,l]); print(f"state {l}: cos mean {c.mean():.5f} min {c.min():.4f}")
k5=np.setdiff1d(np.arange(1536),L5_rms)
dA=Me3[:,5][:,k5]-M[:,5][:,k5]
print("L5 MLP (kept) act corr %.4f, rel change ||d||/||a|| %.3f"%(np.corrcoef(Me3[:,5][:,k5].ravel(),M[:,5][:,k5].ravel())[0,1], np.linalg.norm(dA)/np.linalg.norm(M[:,5][:,k5])))
for l in range(5): print(f"L{l} MLP@CLS corr {np.corrcoef(Me3[:,l].ravel(),M[:,l].ravel())[0,1]:.5f}")
d3=se3-so; print("score diff within-q std %.3f"%qcenter(d3[:,None]).std())
