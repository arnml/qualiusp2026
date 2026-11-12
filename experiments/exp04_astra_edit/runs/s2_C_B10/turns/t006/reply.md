```python
# Hypothesis: low-energy MLP neurons can be removed while preserving high-magnitude and gold-discriminative channels.
apply_config({"layers": [], "heads": [], "neurons": {}})

neuron_order = {}
for layer in range(6):
    stats = neuron_stats[layer]
    e = stats["energy"] / np.median(stats["energy"])
    w = stats["weight_size"] / np.median(stats["weight_size"])
    c = stats["contrast"] / max(float(np.median(stats["contrast"])), 1e-8)
    if layer < 4:
        # Earlier neurons also serve non-CLS tokens: weight magnitude guards against CLS-only inactivity.
        importance = w ** 0.75 * np.maximum(e, .25) ** 0.25
        importance *= np.maximum(1.0, c / 4.0) ** 0.25
    elif layer == 4:
        importance = w ** 0.4 * np.maximum(e, .1) ** 0.6
        importance *= np.maximum(1.0, c / 4.0) ** 0.25
    else:
        # Only this layer's CLS output reaches the scoring head.
        importance = np.maximum(e, .5 * c)
    neuron_order[layer] = np.argsort(importance).tolist()

counts = [384, 384, 384, 512, 640, 1152]
configs["mlp_A"] = {
    "layers": [], "heads": [],
    "neurons": {layer: neuron_order[layer][:counts[layer]] for layer in range(6)}
}
for layer, count in enumerate(counts):
    selected = configs["mlp_A"]["neurons"][layer]
    e = neuron_stats[layer]["energy"]
    print("Layer", layer, "remove", count,
          "fraction of summed squared CLS contribution proxies",
          round(float(np.sum(e[selected] ** 2) / np.sum(e ** 2)), 5))

with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nLayer 1 ablation rejected: 14 paired losses and 2 gains; all layers restored. "
            "Small CLS change did not imply dispensability, consistent with useful token processing.\n"
            "MLP A uses low input/output weight magnitude combined with low CLS output-energy proxies. "
            "Earlier layers emphasize weight magnitude to protect neurons active away from CLS. "
            "Large gold–distractor activation contrasts receive additional protection. "
            "Last-layer pruning emphasizes CLS energy and contrast because no later token mixing occurs. "
            "No compensating scaling is applied.\n")
apply_config(configs["mlp_A"])
mlp_trial = forward(items=discovery_items, model="both")
report_trial("mlp_A", mlp_trial)
```
