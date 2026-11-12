# Hypothesis: preserving more capacity in the late relevance-forming MLP while moderately pruning earlier layers will retain ranking quality.
counts_C = [512, 512, 512, 640, 640, 1280]
configs["mlp_C"] = {
    "layers": [], "heads": [],
    "neurons": {layer: neuron_order[layer][:counts_C[layer]] for layer in range(6)}
}
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nNorm compensation did not recover discovery quality (109/120); discard all scaling. "
            "MLP C backs off pruning relative to B, particularly in layer 4, where the original "
            "CLS representation begins strong alignment with the static classifier direction. "
            "Counts [512,512,512,640,640,1280], no heads/layers removed and no scaling.\n")
apply_config(configs["mlp_C"])
mlp_C_trial = forward(items=discovery_items, model="both")
report_trial("mlp_C", mlp_C_trial)
