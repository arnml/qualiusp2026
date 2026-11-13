# Hypothesis: with the practice margin distribution, within-q score noise of ~0.3 costs <0.5pp in expectation, while ~0.55 costs ~1pp -> sets the perturbation budget for the final config.
rng2=np.random.default_rng(1)
for s in [0.25,0.35,0.45,0.55]:
    nets=[]
    for _ in range(400):
        sn=so+rng2.normal(0,s,size=so.shape); hn=bg3(sn,range(200)); nets.append(hn.mean()-ho.mean())
    nets=np.array(nets); print(f"noise {s}: mean change {100*nets.mean():+.2f}pp  P(change<-1pp) {np.mean(nets<-0.01):.2f}  P(<-0.5pp) {np.mean(nets<-0.005):.2f}")
