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
