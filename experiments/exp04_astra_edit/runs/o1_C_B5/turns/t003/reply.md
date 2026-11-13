Next I'll build the evaluation helpers (both-gold@3 per question, plus within-question rank fidelity) and get baseline numbers per half.

```python
# Hypothesis: baseline both-gold@3 on practice is ~94%; set up metric helpers for paired comparison.
idx={it:k for k,it in enumerate(items_order)}
def bg3(scores, qs):
    hits=[]
    for qi in qs:
        q=practice[qi]; s=np.array([scores[idx[(qi,pi)]] for pi in range(len(q['paragraphs']))])
        top=np.argsort(-s)[:3]; g=np.array(q['gold'])
        hits.append(int(g[top].sum()==g.sum()))
    return np.array(hits)
so=R1['scores_original']
hA=bg3(so,sorted(A)); hB=bg3(so,sorted(B)); hall=bg3(so,range(200))
print("orig bg3 A %.3f B %.3f all %.3f"%(hA.mean(),hB.mean(),hall.mean()))
# gold margin: 2nd-best gold score minus 3rd-best distractor... define margin = min gold - 2nd max distractor? both-gold@3 holds iff min(gold) > 2nd largest distractor
def margins(scores, qs):
    m=[]
    for qi in qs:
        q=practice[qi]; s=np.array([scores[idx[(qi,pi)]] for pi in range(len(q['paragraphs']))]); g=np.array(q['gold'])==1
        d=np.sort(s[~g])[::-1]; m.append(s[g].min()-(d[1] if len(d)>1 else -1e9))
    return np.array(m)
mo=margins(so,range(200)); print("score stats", so.mean(), so.std(), "gold mean", np.mean([so[idx[(qi,pi)]] for qi in range(200) for pi,g in enumerate(practice[qi]['gold']) if g]))
print("margin quantiles", np.quantile(mo[mo>-1e8],[0,.05,.1,.25,.5]))
print("questions with margin in (-1,1):", np.sum(np.abs(mo)<1))
```