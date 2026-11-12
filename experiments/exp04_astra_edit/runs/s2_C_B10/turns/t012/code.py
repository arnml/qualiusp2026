# Hypothesis: the heavier pruning failed mainly because it removed too many layer-4 neurons, so restoring that layer may permit greater earlier sparsity.
counts_D = [640, 640, 640, 768, 640, 1280]
configs["mlp_D"] = {
    "layers": [], "heads": [],
    "neurons": {layer: neuron_order[layer][:counts_D[layer]] for layer in range(6)}
}
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nMLP C: 112/120, one loss and two gains, 28.566528 GFLOPs/question; eligible candidate. "
            "MLP D tests localization of B's damage: retain B's earlier pruning but restore layer 4 "
            "to the C/A count of 640 removed neurons. All rankings and importance measures stay fixed.\n")
apply_config(configs["mlp_D"])
mlp_D_trial = forward(items=discovery_items, model="both")
report_trial("mlp_D", mlp_D_trial)
