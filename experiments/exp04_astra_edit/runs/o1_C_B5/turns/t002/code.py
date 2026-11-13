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
