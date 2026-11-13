Pass 3 diagnostics:
- **Early silent-neuron removal is nearly invisible at [CLS]** (cos 0.9991 at layer 4's output).
- **Removing layer-5 heads 11 and 5 moved the layer-5 MLP input by ~13%**, yet ranking held. So layer-5 heads are not free, but some can go.

With 2 passes left, I'll climb a ladder:
- **Pass 4:** a bolder superset of pass 3.
- **Pass 5:** go further if pass 4 is clean, otherwise take a smaller step.
- **Fallback:** pass 3, which is tested clean.

To pick early-layer heads for pass 4, I'll score each head in layers 0–4 on "no-op" evidence: attention-sink mass (CLS/SEP/punctuation), OV norm, and role (positional, lexical, or cross-segment heads are excluded).

```python
# Hypothesis: a few early heads combine high sink mass with low OV norm and no identifiable positional/lexical/cross-segment role -> best no-op candidates.
sink=res[...,0]+res[...,1]+res[...,2]
rows=[]
for l in range(5):
    for h in range(12):
        prev_,next_,self_,pq,qp,st=stats[l,h]
        role=('POS' if max(prev_,next_)>0.2 else '')+('LEX' if st>0.1 else '')+('XSEG' if pq>0.15 else '')
        rows.append((l,h,sink[l,h],ov[l,h],ro[l,h] if l>=3 else np.nan,role))
import pandas as pd
df=pd.DataFrame(rows,columns=['L','H','sink','OV','readoutOV','role'])
df['noop']=df.sink/df.OV
print(df[df.role==''].sort_values('noop',ascending=False).head(16).round(3).to_string(index=False))
```