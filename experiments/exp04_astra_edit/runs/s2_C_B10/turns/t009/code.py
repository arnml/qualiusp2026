# Hypothesis: the MLP has more removable capacity than attention, so increasing neuron pruning may save more with less ranking damage.
counts_B = [640, 640, 640, 768, 896, 1280]
configs["mlp_B"] = {
    "layers": [], "heads": [],
    "neurons": {layer: neuron_order[layer][:counts_B[layer]] for layer in range(6)}
}
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nEight-head combination rejected: 102/120, nine losses against the original. "
            "The extra savings did not justify its quality damage. Restore all heads, "
            "and test greater MLP sparsity using the same fixed diagnostic ranking. "
            "MLP B removal counts: [640,640,640,768,896,1280].\n")
apply_config(configs["mlp_B"])
mlp_B_trial = forward(items=discovery_items, model="both")
report_trial("mlp_B", mlp_B_trial)
