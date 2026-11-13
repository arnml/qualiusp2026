The conservative fallback preserves every original success: 112/120 on selection versus 111/120, and 78/80 on confirmation for both models. It saves 12.4% of encoder FLOPs. I’m freezing it now and using the remaining passes to check why the retained components matter. Since confirmation prompted the rollback, its result is no longer an untouched estimate.


```python
# Hypothesis: the retained final-layer neurons carry task signal, while the removed channels have low activation-weight salience.
final_label = "conservative_early96_late90"
final_cfg = configs[final_label]
apply_config(**final_cfg)

for l in range(6):
    removed = np.asarray(final_cfg["neurons"][l], dtype=int)
    imp = neuron_importance[l]
    rms_energy_share = float(np.sum(imp[removed]**2) / np.sum(imp**2))
    print("Layer", l, "removed", len(removed), "remaining", 1536-len(removed),
          "removed share of summed CLS channel energy proxy:", round(rms_energy_share, 6))
with NOTES.open("a") as f:
    f.write("\nFINAL CONFIGURATION FROZEN: conservative_early96_late90. "
            "Remove bottom 96 weight-salience neurons in each layer 0–4 and bottom 1382 "
            "CLS activation-times-output-norm neurons in layer 5. Keep all heads and layers; no rescaling. "
            "Selection: 112/120 versus 111/120; no original successes lost. "
            "Confirmation: 78/80 versus 78/80; no paired losses or gains. "
            "Confirmation prompted conservative rollback, so it is not an untouched final estimate. "
            "Remaining passes are mechanistic controls on selection data and final verification, not tuning.\n")

control_configs = {}
control_configs["remove_all_final_MLP"] = {
    "neurons": {l: (list(range(1536)) if l == 5 else list(final_cfg["neurons"][l])) for l in range(6)},
    "heads": [], "layers": [],
}
control_configs["remove_high_strength_early_channels"] = {
    "neurons": {l: (weight_ranks[l][-96:].tolist() if l < 5 else list(final_cfg["neurons"][l])) for l in range(6)},
    "heads": [], "layers": [],
}
for label, cfg in control_configs.items():
    apply_config(**cfg)
    out = forward(items=select_items, model="both")
    report("MECHANISTIC_CONTROL_" + label, out)
    apply_config(**final_cfg)
print("Frozen final model restored:", edited())
```
