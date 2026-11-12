The eight-head edit hurt ranking quality, so I restored all heads. Heavier MLP pruning cuts cost by 32%, but loses two net discovery successes and substantially shrinks margins. I’ll test a fixed, norm-based rescaling of surviving MLP columns to see whether it compensates for some of that loss.


```python
# Hypothesis: fixed norm-based rescaling of ordinary surviving MLP channels can compensate for amplitude lost through pruning.
configs["mlp_B_scaled"] = copy.deepcopy(configs["mlp_B"])
configs["mlp_B_scaled"]["scales"] = []
for layer in range(6):
    stats = neuron_stats[layer]
    protected = set(np.argsort(stats["energy"])[-16:].tolist())
    removed = set(configs["mlp_B"]["neurons"][layer])
    regular = np.array([i for i in range(1536) if i not in protected], dtype=int)
    surviving_regular = np.array([i for i in regular if i not in removed], dtype=int)
    proxy = stats["weight_size"] if layer < 4 else stats["energy"]
    factor = min(1.25, float(np.sqrt(np.sum(proxy[regular] ** 2) /
                                    np.sum(proxy[surviving_regular] ** 2))))
    configs["mlp_B_scaled"]["scales"].append(
        (f"bert.encoder.layer.{layer}.output.dense.weight", surviving_regular.tolist(), factor)
    )
    print("Layer", layer, "scale", round(factor, 4),
          "surviving regular columns", len(surviving_regular),
          "protected high-energy columns", len(protected))

with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nMLP B: 109/120 versus 111/120, three losses and one gain; unscaled version "
            "fails the discovery one-point requirement. Test a prespecified energy-compensation rule: "
            "sqrt(total proxy squared norm / retained proxy squared norm), capped at 1.25, "
            "using weight products in layers 0–3 and observed CLS energy in layers 4–5. "
            "The 16 highest-energy columns per layer are not scaled, to protect dominant channels. "
            "Factors are statistics, not fitted to scores, labels, or a loss.\n")
apply_config(configs["mlp_B_scaled"])
scaled_trial = forward(items=discovery_items, model="both")
report_trial("mlp_B_scaled", scaled_trial)
```
