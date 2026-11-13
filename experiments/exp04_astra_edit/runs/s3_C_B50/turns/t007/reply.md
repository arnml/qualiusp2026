```python
# Hypothesis: weight-based MLP salience protects features used at paragraph tokens that CLS-only activity can miss.
weight_ranks = {}
weight_importance = {}
for l in range(6):
    wi = arr(load_original(wn(l, "intermediate.dense.weight")))
    bi = arr(load_original(wn(l, "intermediate.dense.bias")))
    wo = arr(load_original(wn(l, "output.dense.weight")))
    strength = np.sqrt(np.sum(wi**2, axis=1) + bi**2) * np.linalg.norm(wo, axis=0)
    weight_importance[l] = strength
    weight_ranks[l] = np.argsort(strength)

for fraction in [.25, .40]:
    label = f"MLP_weight_{int(fraction*100)}"
    nc = {l: weight_ranks[l][:int(1536 * fraction)].tolist() for l in range(6)}
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    report(label, out)
reset_all()
```
