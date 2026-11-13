# Hypothesis: removing the least active quarter of each MLP will preserve ranking better than removing an entire early layer.
def apply_config(neurons=None, heads=(), layers=()):
    reset_all()
    for l in layers:
        remove_layer(int(l))
    for l, h in heads:
        remove_head(int(l), int(h))
    for l, ids in (neurons or {}).items():
        if len(ids):
            remove_neurons(int(l), [int(i) for i in ids])

configs = {}
outputs = {}
for fraction in [.25, .50]:
    label = f"MLP_CLS_{int(fraction*100)}"
    nc = {l: neuron_ranks[l][:int(1536 * fraction)].tolist() for l in range(6)}
    configs[label] = {"neurons": nc, "heads": [], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    report(label, out)
reset_all()
with NOTES.open("a") as f:
    f.write("\nWhole-layer removals rejected: layer 1 loses 11/120 net and layer 2 loses 4/120 net; "
            "small residual updates are necessary. Testing structured MLP pruning next.\n")
