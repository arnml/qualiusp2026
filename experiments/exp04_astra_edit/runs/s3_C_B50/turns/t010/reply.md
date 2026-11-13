```python
# Hypothesis: the final MLP has especially low-use channels, and safe layer-specific cuts may combine into a substantially cheaper encoder.
specs = {
    "MLP_late75": [384, 384, 384, 384, 384, 1152],
    "MLP_mixed": [384, 768, 384, 384, 768, 768],
    "MLP_mixed_deep": [384, 768, 384, 768, 768, 1152],
}
for label, counts in specs.items():
    nc = {l: (neuron_ranks[l] if l == 5 else weight_ranks[l])[:counts[l]].tolist()
          for l in range(6)}
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    report(label, out)
reset_all()
```
