# Hypothesis: P3 + (L5 RMS-1000, L5 head 8, early max|a|<0.02 neurons, heads L4h9/L1h6/L1h7) keeps within-q score-diff std < ~0.4 and no net bg3 loss.
reset_all()
L5_rms1000=[int(j) for j in oR[:1000]]; remove_neurons(5,L5_rms1000)
for h in [11,5,8]: remove_head(5,h)
silent2={l:[int(j) for j in np.where(np.abs(M[:,l]).max(0)<0.02)[0]] for l in range(4)}
for l,js in silent2.items(): remove_neurons(l,js)
for (l,h) in [(4,9),(1,6),(1,7)]: remove_head(l,h)
print({l:len(v) for l,v in silent2.items()}, cost())
R4=forward(items=items_order, model="edited", cls=True, mlp=True)
pickle.dump(R4, open(WORK/"pass4.pkl","wb"))
se4=np.asarray(R4['scores_edited']); report(se4)
d4=se4-so; print("score diff mean %+.2f |diff| %.2f within-q std %.3f"%(d4.mean(), np.abs(d4).mean(), qcenter(d4[:,None]).std()))
Ce4=np.asarray(R4['cls_edited'],np.float32)
print("CLS cos per state P4:", [round(float(cos(Ce4[:,l],C[:,l]).mean()),5) for l in range(7)])
