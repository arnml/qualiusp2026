```python
# Hypothesis: early token processing needs more capacity than the middle MLPs; weak layer-1 heads may provide additional safe savings.
for label, counts in {
    "MLP_balanced2": [256, 768, 256, 512, 768, 1382],
    "MLP_balanced3": [192, 896, 192, 384, 896, 1382],
}.items():
    nc = {l: (neuron_ranks[l] if l == 5 else weight_ranks[l])[:counts[l]].tolist()
          for l in range(6)}
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    report(label, out)

label = "balanced_heads1"
configs[label] = {"neurons": configs["MLP_balanced"]["neurons"],
                  "heads": [(1, h) for h in head_ranks[1][:3]], "layers": []}
apply_config(**configs[label])
out = forward(items=select_items, model="both")
outputs[label] = np.asarray(out["scores_edited"]).copy()
report(label, out)
reset_all()
```
