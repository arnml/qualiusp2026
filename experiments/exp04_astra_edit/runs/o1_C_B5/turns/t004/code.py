# Hypothesis: several heads (esp. middle/late layers) park attention on [SEP]/[CLS]/punctuation and act as near no-ops; last-layer heads matter only via the CLS row.
att=R1['attn_original']; toks=R1['attn_tokens_original']
import string
res=np.zeros((6,12,5))  # cls, sep, punct, self(diag), cross-segment(only CLS row later)
clsrow=np.zeros((6,12,4)) # CLS-row: mass on cls, sep, punct, question tokens
for n in range(8):
    t=list(toks[n]); L=sum(1 for x in t if x not in ('[PAD]','',None))
    t=t[:L]; iscls=np.array([x=='[CLS]' for x in t]); issep=np.array([x=='[SEP]' for x in t])
    ispun=np.array([all(c in string.punctuation for c in x) for x in t])
    sep1=np.where(issep)[0][0]; isq=np.zeros(L,bool); isq[1:sep1]=True
    A_=att[n][:,:,:L,:L]
    res[...,0]+=A_[...,iscls].sum(-1).mean(-1)/8; res[...,1]+=A_[...,issep].sum(-1).mean(-1)/8
    res[...,2]+=A_[...,ispun].sum(-1).mean(-1)/8; res[...,3]+=np.einsum('lhii->lh',A_)/L/8
    r=A_[:,:,0,:]; clsrow[...,0]+=r[...,iscls].sum(-1)/8; clsrow[...,1]+=r[...,issep].sum(-1)/8; clsrow[...,2]+=r[...,ispun].sum(-1)/8; clsrow[...,3]+=r[...,isq].sum(-1)/8
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("tokens example:", toks[0][:20])
for l in range(6):
    print(f"L{l} all-queries  CLS+SEP+punct mass:", (res[l,:,0]+res[l,:,1]+res[l,:,2]))
for l in range(6):
    print(f"L{l} CLS-row: special(CLS+SEP)", clsrow[l,:,0]+clsrow[l,:,1], " question", clsrow[l,:,3])
