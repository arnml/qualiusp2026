```python
# Hypothesis: a separator-focused weak head or small fixed MLP rescaling may improve the best layer-specific pruning without harming ranking.
label = "balanced2_head4_9"
configs[label] = {"neurons": configs["MLP_balanced2"]["neurons"], "heads": [(4, 9)], "layers": []}
apply_config(**configs[label])
out = forward(items=select_items, model="both")
outputs[label] = np.asarray(out["scores_edited"]).copy()
report(label, out)

label = "balanced2_scale1.1"
apply_config(**configs["MLP_balanced2"])
for l in range(6):
    kept = sorted(set(range(1536)) - set(configs["MLP_balanced2"]["neurons"][l]))
    scale_cols(wn(l, "output.dense.weight"), kept, 1.10)
out = forward(items=select_items, model="both")
outputs[label] = np.asarray(out["scores_edited"]).copy()
report(label, out)

label = "balanced2_earlyprotected"
counts = [192, 768, 192, 512, 896, 1382]
configs[label] = {
    "neurons": {l: (neuron_ranks[l] if l == 5 else weight_ranks[l])[:counts[l]].tolist()
                for l in range(6)},
    "heads": [], "layers": [],
}
apply_config(**configs[label])
out = forward(items=select_items, model="both")
outputs[label] = np.asarray(out["scores_edited"]).copy()
report(label, out)
reset_all()
```
