# Hypothesis: (1) on questions least damaged before layer 5, within-q ranking is still mostly intact -> layer-5 damage is mainly an offset; (2) ranking L5 neurons by RMS contribution gives sets with much smaller total perturbation.
from scipy.stats import spearmanr
qc4=np.array([np.mean([c4[idx[(qi,p)]] for p in range(len(practice[qi]['paragraphs']))]) for qi in range(200)])
rho_q=np.array([spearmanr([so[idx[(qi,p)]] for p in range(10)],[se[idx[(qi,p)]] for p in range(10)])[0] if len(practice[qi]['paragraphs'])==10 else np.nan for qi in range(200)])
o_q=np.argsort(-qc4)
for lo,hi in [(0,40),(40,80),(80,120),(120,160),(160,200)]:
    s=o_q[lo:hi]; print(f"questions ranked {lo}-{hi} by intactness (mean L4cos {qc4[s].mean():.3f}): spearman {np.nanmean(rho_q[s]):.3f}")
ga5=W('bert.encoder.layer.5.attention.output.LayerNorm.weight'); ba5=W('bert.encoder.layer.5.attention.output.LayerNorm.bias')
ref=np.sqrt((ga5**2).sum()+(ba5**2).sum()); print("L5 LN_a output scale ref ||x_a|| ~ %.1f"%ref)
wn=np.linalg.norm(Wout5,axis=0); rmsA=np.sqrt((a5[inA]**2).mean(0))*wn
oR=np.argsort(rmsA)
for k in [400,600,800,1000,1300]:
    for nm,S in [('RMS',oR[:k]),('var',oA[:k])]:
        D=a5[:, S]@Wout5[:,S].T   # per-item removed contribution vector (attribution, not a forward pass)
        print(f"k={k} {nm}: per-item ||removed|| mean {np.linalg.norm(D,axis=1).mean():.2f} ({100*np.linalg.norm(D,axis=1).mean()/ref:.1f}% of ref)  within-q proj std {qcenter_mask((a5[:,S]*proj[S]).sum(1)[:,None],inA).std():.3f}")
