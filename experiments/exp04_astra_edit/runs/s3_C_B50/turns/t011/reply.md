```python
# Hypothesis: heads with the smallest value-to-output weight strength contribute little signal and can be removed in layer-local groups.
head_ranks = {}
head_group_results = {}
head_parent = "MLP_late75"
for l in range(6):
    v = arr(load_original(wn(l, "attention.self.value.weight"))).reshape(12, 32, 384)
    vb = arr(load_original(wn(l, "attention.self.value.bias"))).reshape(12, 32)
    o = arr(load_original(wn(l, "attention.output.dense.weight"))).reshape(384, 12, 32)
    strength = np.sqrt(np.sum(v**2, axis=(1, 2)) + np.sum(vb**2, axis=1)) * np.sqrt(np.sum(o**2, axis=(0, 2)))
    head_ranks[l] = np.argsort(strength).tolist()
    hs = [(l, h) for h in head_ranks[l][:3]]
    label = f"{head_parent}_heads_layer{l}"
    configs[label] = {"neurons": configs[head_parent]["neurons"], "heads": hs, "layers": []}
    print("Testing heads:", hs)
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    head_group_results[l] = report(label, out)
reset_all()
```
