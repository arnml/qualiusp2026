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
