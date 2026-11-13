The [CLS]-safe neuron sets in early layers are small: about 280 neurons have max |output| below 0.02 in layers 0–3, worth only ~0.7 GFLOPs. Layer-5 neurons are never fully off, but their direct effect on the decision is small. Real savings need head removals or a whole layer.

To judge which heads (or which layer) carry the question–paragraph matching this task depends on, I'll characterize each head's attention behavior: previous/next-token, self, cross-segment (paragraph↔question), and attention to identical tokens elsewhere (lexical matching).

```python
# Hypothesis: task-critical heads do cross-segment lexical matching (paragraph token -> same token in question); positional heads build local context; layers whose heads are mostly diffuse/self may be skippable.
stats=np.zeros((6,12,6))  # prev, next, self, p->q cross, q->p cross, same-token(other pos)
for n in range(8):
    t=[str(x) for x in toks[n]]; L=Ls[n]; t=t[:L]
    seps=[i for i,x in enumerate(t) if x=='[SEP]']; s1=seps[0]
    seg=np.zeros(L,int); seg[s1+1:]=1
    real=np.array([x not in ('[CLS]','[SEP]') for x in t])
    A_=att[n][:,:,:L,:L]
    prev=np.mean([A_[:,:,i,i-1] for i in range(1,L)],0); nxt=np.mean([A_[:,:,i,i+1] for i in range(L-1)],0)
    selfm=np.einsum('lhii->lh',A_)/L
    pq=A_[:,:,(seg==1)&real][:,:,:,(seg==0)&real].sum(-1).mean(-1)
    qp=A_[:,:,(seg==0)&real][:,:,:,(seg==1)&real].sum(-1).mean(-1)
    same=np.array([[x==y and i!=j and x not in ('[CLS]','[SEP]','.',',','the','of','a','(',')') for j,y in enumerate(t)] for i,x in enumerate(t)])
    has=same.any(1)&real
    st=(A_[:,:,has,:]*same[has][None,None]).sum(-1).mean(-1) if has.any() else 0
    stats+=np.stack([prev,nxt,selfm,pq,qp,st],-1)/8
names=['prev','next','self','p->q','q->p','same-tok']
for l in range(6):
    print(f"--- L{l}"); 
    for k,nm in enumerate(names): print(f"  {nm:8s}", stats[l,:,k].round(2))
```