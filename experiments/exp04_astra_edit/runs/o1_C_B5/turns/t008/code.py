# Hypothesis: late heads differ in how much their OV circuit can write onto the decision direction; early-layer neuron activity can be proxied from weights (bias vs input-weight scale).
print("within-q std of final h.d6:", qcenter((C[:,6]@d6)[:,None]).std().round(3), " (MLP5 contribution pre-LN, /sigma unknown)")
ro=np.zeros((6,12))
for l in [3,4,5]:
    Wv=W(f'bert.encoder.layer.{l}.attention.self.value.weight'); Wo=W(f'bert.encoder.layer.{l}.attention.output.dense.weight')
    ga=W(f'bert.encoder.layer.{l}.attention.output.LayerNorm.weight'); go=W(f'bert.encoder.layer.{l}.output.LayerNorm.weight')
    dl=qcenter(C[:,l+1]); dl=dl[inA&(gold==1)].mean(0)-dl[inA&(gold==0)].mean(0); dl/=np.linalg.norm(dl)
    r=ga*go*dl
    for h in range(12): ro[l,h]=np.linalg.norm(r@Wo[:,32*h:32*h+32]@Wv[32*h:32*h+32,:])
for l in [3,4,5]: print(f"L{l} readout-OV:", ro[l].round(2), " rank:", np.argsort(ro[l]))
# weight-based activity proxy for neurons: z = w.y + b with y=LN output (gamma*n+beta), n ~ unit-variance isotropic
for l in range(6):
    Wi=W(f'bert.encoder.layer.{l}.intermediate.dense.weight'); bi=W(f'bert.encoder.layer.{l}.intermediate.dense.bias')
    ga=W(f'bert.encoder.layer.{l}.attention.output.LayerNorm.weight'); ba=W(f'bert.encoder.layer.{l}.attention.output.LayerNorm.bias')
    mu=Wi@ba+bi; sd=np.linalg.norm(Wi*ga[None,:],axis=1)
    z=mu/sd
    # compare with CLS reality
    pre_active=(M[:,l]>0.01).mean(0)
    print(f"L{l}: proxy z quantiles", np.quantile(z,[.1,.5,.9]).round(2), " corr(z, CLS active frac)=%.2f"%np.corrcoef(z,pre_active)[0,1])
