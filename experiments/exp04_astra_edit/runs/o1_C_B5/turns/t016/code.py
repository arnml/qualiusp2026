# Hypothesis: final damage tracks layer-4 output damage closely; the extra damage from layer-5 edits is modest (check via binning).
print("c4 quantiles", np.quantile(c4,[0,.1,.5,.9,1]).round(3))
bins=np.quantile(c4,[0,.2,.4,.6,.8,1.0])
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(c4>=lo)&(c4<=hi)
    print(f"L4cos [{lo:.3f},{hi:.3f}]: n={m.sum()} final cos {c5[m].mean():.3f}  |dscore| {np.abs(ds[m]).mean():.2f}  mean dscore {ds[m].mean():+.2f}")
print("all: |dscore| %.2f mean %+.2f; score std %.2f"%(np.abs(ds).mean(),ds.mean(),so.std()))
