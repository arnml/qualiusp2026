The final MLP tolerates a 75% cut to its least active channels, alongside 25% cuts elsewhere, with no lost selection-set successes. Attention pruning is more sensitive: several groups of weak heads caused regressions. I’ll test the promising cuts together, since individually safe removals can interact.


```python
# Hypothesis: preserving more early-layer capacity while concentrating pruning in the final MLP will improve the cost–quality tradeoff.
new_specs = {
    "MLP_late90": [384, 384, 384, 384, 384, 1382],
    "MLP_balanced": [256, 640, 256, 384, 640, 1382],
    "MLP_mixed_late90": [384, 768, 384, 384, 768, 1382],
}
for label, counts in new_specs.items():
    nc = {l: (neuron_ranks[l] if l == 5 else weight_ranks[l])[:counts[l]].tolist()
          for l in range(6)}
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    report(label, out)
reset_all()
```
