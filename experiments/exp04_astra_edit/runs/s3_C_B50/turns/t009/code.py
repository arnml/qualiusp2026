# Hypothesis: MLP redundancy varies by layer, so deeper cuts should target layers that tolerate them without paired ranking regressions.
mlp_layer_results = {}
for l in range(6):
    label = f"MLP_base25_layer{l}_50"
    nc = {k: list(v) for k, v in configs["MLP_weight_25"]["neurons"].items()}
    rank = neuron_ranks[l] if l == 5 else weight_ranks[l]
    nc[l] = rank[:768].tolist()
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    mlp_layer_results[l] = report(label, out)
reset_all()
