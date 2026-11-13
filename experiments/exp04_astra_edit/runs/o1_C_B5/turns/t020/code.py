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
