# Hypothesis: layer-5 neurons differ hugely in how much their CLS output moves the gold-vs-distractor direction; most are negligible.
def W(n): return load_original(n).numpy()
d6=None
Xc=qcenter(C[:,6]); d6=Xc[inA&(gold==1)].mean(0)-Xc[inA&(gold==0)].mean(0); d6/=np.linalg.norm(d6)
g5=W('bert.encoder.layer.5.output.LayerNorm.weight')
Wout5=W('bert.encoder.layer.5.output.dense.weight')  # (384,1536)
proj=(Wout5*(g5*d6)[:,None]).sum(0)     # per-neuron effect on d6 per unit act (pre-LN, up to 1/sigma)
a5=M[:,5]
# within-question std of each neuron's projected contribution, and mean contribution
aqc=qcenter(a5); wq_std=aqc.std(0); mean_a=a5.mean(0)
imp_var=wq_std*np.abs(proj); imp_mean=np.abs(mean_a*proj); shift_norm=np.abs(mean_a)*np.linalg.norm(Wout5*g5[:,None],axis=0)
tot_var=np.sqrt((qcenter(a5*proj).sum(1)**2).mean()) # within-q std of total MLP projected contribution
print("total MLP5 within-q projected std:", tot_var.round(3))
o=np.argsort(imp_var); cs=np.cumsum(imp_var[o])
for k in [500,768,1000,1200,1300,1400]:
    S=o[:k]; dv=qcenter((a5[:,S]*proj[S]).sum(1)); 
    print(k, "removed-set within-q proj std %.3f (%.1f%% of total)"%(dv.std(), 100*dv.std()/tot_var), " mean-shift norm %.2f"%np.linalg.norm((mean_a[S]*Wout5[:,S]*g5[:,None]).sum(1)))
# head OV norms per layer
ov=np.zeros((6,12))
for l in range(6):
    Wv=W(f'bert.encoder.layer.{l}.attention.self.value.weight'); Wo=W(f'bert.encoder.layer.{l}.attention.output.dense.weight')
    for h in range(12): ov[l,h]=np.linalg.norm(Wo[:,32*h:32*h+32]@Wv[32*h:32*h+32,:])
print("OV norms:\n", ov.round(1))
