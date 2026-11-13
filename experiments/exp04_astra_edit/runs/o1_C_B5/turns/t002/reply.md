Cost is linear: each head is about 0.204 GFLOPs (0.52%), each MLP neuron about 0.0026 GFLOPs, and a layer is about 6.46 GFLOPs (16.5%). Neurons make up about 61% of the total and heads about 37%.

**Plan for the 5 passes**
1. Original model on all 1,992 items, with CLS states, CLS MLP activations, and attention for 8 diverse inputs (diagnosis).
2. through 4. Edited candidates on all items, read on the selection half.
5. Final confirmation.

I'm splitting the questions with a fixed random permutation: 100 for selection (A) and 100 held out for confirmation (B).

```python
# Hypothesis: one full diagnostic pass of the original model (scores + CLS states + MLP@CLS + attention on 8 diverse inputs) is enough to rank candidates.
rng=np.random.default_rng(0); perm=rng.permutation(200); A=set(perm[:100].tolist()); B=set(perm[100:].tolist())
items=[(qi,pi) for qi,q in enumerate(practice) for pi in range(len(q['paragraphs']))]
# put 8 diverse inputs first for attention: 4 questions from A, one gold + one distractor each
first=[]
for qi in sorted(A)[:4]:
    g=practice[qi]['gold']; first.append((qi,g.index(1))); first.append((qi,g.index(0)))
rest=[it for it in items if it not in first]; items_order=first+rest
R1=forward(items=items_order, model="original", cls=True, mlp=True, attn_n=8)
print({k:(v.shape if hasattr(v,'shape') else type(v)) for k,v in R1.items()})
import pickle; pickle.dump((items_order,R1,sorted(A),sorted(B)), open(WORK/"pass1.pkl","wb"))
```