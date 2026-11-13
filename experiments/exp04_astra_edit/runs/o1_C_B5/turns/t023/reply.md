Localization is only partly conclusive.

Per-item score changes barely track pre-layer-5 [CLS] perturbation (r ≈ 0.1). The layer-by-layer MLP-input correlations show where the extra pass-4 perturbation is injected:

| Layer | Pass 3 (1−corr) | Pass 4 (1−corr) | Ratio |
|---|---|---|---|
| L1 | 9e-5 | 7.7e-4 | 8.6× |
| L2 | 8.6e-4 | 2.4e-3 | 2.8× |
| L3 | 2.2e-3 | 8.8e-3 | 4× |
| L4 | 4.9e-3 | 2.3e-2 | 4.8× |

The big jump at layer 1 points at the layer-1 heads 6 and 7 (plus extra layer-0/1 neurons). The growth from layer 3 to 4 is similar in both passes (2.2× vs 2.6×), so L4h9 adds little beyond propagation. The only new flip is question 13 (split A); split B is unchanged in both passes.

Before spending the last pass, I'll calibrate how much score noise the metric tolerates. I'll add Gaussian noise of a given within-question std to the original practice scores. This is a statistical noise model on outputs, not a network computation.

```python
# Hypothesis: with the practice margin distribution, within-q score noise of ~0.3 costs <0.5pp in expectation, while ~0.55 costs ~1pp -> sets the perturbation budget for the final config.
rng2=np.random.default_rng(1)
for s in [0.25,0.35,0.45,0.55]:
    nets=[]
    for _ in range(400):
        sn=so+rng2.normal(0,s,size=so.shape); hn=bg3(sn,range(200)); nets.append(hn.mean()-ho.mean())
    nets=np.array(nets); print(f"noise {s}: mean change {100*nets.mean():+.2f}pp  P(change<-1pp) {np.mean(nets<-0.01):.2f}  P(<-0.5pp) {np.mean(nets<-0.005):.2f}")
```