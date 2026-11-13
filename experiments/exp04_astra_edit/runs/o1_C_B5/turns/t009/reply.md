The weight-only activity proxy barely tracks [CLS] activity in layers 0–4 (r = 0.06–0.26), so I won't rely on it. Readout-OV norms point to layer-5 heads 11 and 5, and layer-4 heads 9, 11, 10 and 7, as having little direct write onto the decision direction.

Before building a group-test configuration, I'll check how many neurons per layer are deeply off at [CLS] (|output| near zero on every item). Removing those changes [CLS] by essentially nothing.

```python
# Hypothesis: a sizable set of neurons per layer is 'deeply off' at CLS (|gelu|<~0.005 on all 1992 items) -> zero CLS effect; these are the safest early-layer neuron candidates.
for l in range(6):
    a=M[:,l]; mx=np.abs(a).max(0)
    print(f"L{l}: max|a|<0.005: {(mx<0.005).sum():4d}   <0.02: {(mx<0.02).sum():4d}   <0.05: {(mx<0.05).sum():4d}   near-floor const (std<0.01 & mean<-0.1): {((a.std(0)<0.01)&(a.mean(0)<-0.1)).sum()}")
# for layer 5, how big is the readout effect of removing deeply-off + low-imp sets
o5=np.argsort(imp_var)
print("L5 imp_var of 1300th neuron:", imp_var[o5[1299]], " max:", imp_var.max())
```