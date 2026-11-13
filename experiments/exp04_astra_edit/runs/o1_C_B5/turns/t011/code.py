# Hypothesis: some middle layer (1 or 2) contributes negligibly to the CLS readout direction -> candidate for whole-layer removal; late layers contribute most.
u=W('bert.pooler.dense.weight').T@W('classifier.weight').ravel(); u/=np.linalg.norm(u)
for l in range(6):
    dl=C[:,l+1]-C[:,l]
    pu=dl@u; pd=dl@d6
    wq=qcenter(pu[:,None]).ravel(); wqd=qcenter(pd[:,None]).ravel()
    gcorr=np.corrcoef(wqd,gold)[0,1]
    print(f"layer {l}: |delta CLS| {np.linalg.norm(dl,axis=1).mean():.2f}  within-q std on u {wq.std():.3f}  on d6 {wqd.std():.3f}  corr(d6-proj, gold) {gcorr:+.2f}")
print("final CLS within-q std on u %.3f, on d6 %.3f"%(qcenter((C[:,6]@u)[:,None]).std(), qcenter((C[:,6]@d6)[:,None]).std()))
print("cos(u,d6)=%.2f"%(u@d6))
