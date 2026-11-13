The cell failed on a dtype mismatch, probably because the attention array or the tokens are object-typed. I'll inspect the dtypes and retry.

```python
# Hypothesis: attention array or tokens are object dtype; inspect and cast.
print(att.dtype, toks.dtype, toks[0][:12], toks[0][-5:])
att=np.asarray(att, dtype=np.float32)
res=np.zeros((6,12,4)); clsrow=np.zeros((6,12,4)); Ls=[]
for n in range(8):
    t=[str(x) for x in toks[n]]; L=sum(1 for x in t if x not in ('[PAD]','','None')); Ls.append(L)
    t=t[:L]; iscls=np.array([x=='[CLS]' for x in t]); issep=np.array([x=='[SEP]' for x in t])
    ispun=np.array([all(c in string.punctuation for c in x) for x in t])
    sep1=np.where(issep)[0][0]; isq=np.zeros(L,bool); isq[1:sep1]=True
    A_=att[n][:,:,:L,:L]
    res[...,0]+=A_[...,iscls].sum(-1).mean(-1)/8; res[...,1]+=A_[...,issep].sum(-1).mean(-1)/8
    res[...,2]+=A_[...,ispun].sum(-1).mean(-1)/8; res[...,3]+=np.einsum('lhii->lh',A_)/L/8
    r=A_[:,:,0,:]; clsrow[...,0]+=r[...,iscls].sum(-1)/8; clsrow[...,1]+=r[...,issep].sum(-1)/8; clsrow[...,2]+=r[...,ispun].sum(-1)/8; clsrow[...,3]+=r[...,isq].sum(-1)/8
print("lengths", Ls)
for l in range(6): print(f"L{l} all-q CLS+SEP+punct:", res[l,:,0]+res[l,:,1]+res[l,:,2])
for l in range(6): print(f"L{l} CLS-row special:", clsrow[l,:,0]+clsrow[l,:,1], " q:", clsrow[l,:,3])
```