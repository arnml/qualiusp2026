# Hypothesis: CLS changes most in late layers; layer-5 MLP is sparse at CLS (many neurons ~0), making most of it removable exactly.
C=np.asarray(R1['cls_original'],np.float32); M=np.asarray(R1['mlp_original'],np.float32)
def cos(a,b): return (a*b).sum(-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1)
print("cos(cls_l, cls_l+1):", [round(float(cos(C[:,l],C[:,l+1]).mean()),3) for l in range(6)])
print("cos(cls_l, cls_final):", [round(float(cos(C[:,l],C[:,6]).mean()),3) for l in range(7)])
# discriminability per layer: class-mean-difference direction (gold - distractor, question-centered) fit on A, eval on B (top-2 hit)
gold=np.array([practice[qi]['gold'][pi] for qi,pi in items_order]); qid=np.array([qi for qi,pi in items_order])
inA=np.isin(qid,sorted(A))
def qcenter(X):
    Xc=X.copy()
    for qi in range(200): m=qid==qi; Xc[m]-=X[m].mean(0)
    return Xc
for l in range(7):
    Xc=qcenter(C[:,l]); d=Xc[inA&(gold==1)].mean(0)-Xc[inA&(gold==0)].mean(0)
    s=C[:,l]@d; print(f"layer-out {l}: B bg3 with mean-diff direction = {bg3(s,sorted(B)).mean():.2f}")
# MLP sparsity at CLS
for l in range(6):
    a=M[:,l]; ma=np.abs(a).mean(0)
    print(f"L{l}: mean|act| quantiles", np.quantile(ma,[.1,.25,.5,.75,.9]).round(3), " frac neurons with max act<0.01:", (a.max(0)<0.01).mean().round(3), " frac mean|a|<0.02:", (ma<0.02).mean().round(3))
