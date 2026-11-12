# Hypothesis: unusually small output paths or repeated weights identify inexpensive pruning candidates.
def tensor(layer, suffix):
    return load_original(f"bert.encoder.layer.{layer}.{suffix}").detach().cpu()

def quantiles(x):
    return np.round(np.quantile(np.asarray(x), [0, .1, .25, .5, .75, .9, 1]), 5).tolist()

weight_stats = {}
for layer in range(6):
    wi = tensor(layer, "intermediate.dense.weight")
    bi = tensor(layer, "intermediate.dense.bias")
    wo = tensor(layer, "output.dense.weight")
    ao = tensor(layer, "attention.output.dense.weight")
    q = tensor(layer, "attention.self.query.weight")
    k = tensor(layer, "attention.self.key.weight")
    v = tensor(layer, "attention.self.value.weight")
    head_out = [float(ao[:, 32*h:32*(h+1)].norm()) for h in range(12)]
    head_value_path = [
        float((ao[:, 32*h:32*(h+1)] @ v[32*h:32*(h+1)]).norm())
        for h in range(12)
    ]
    weight_stats[layer] = {
        "mlp_input_norm": wi.norm(dim=1).numpy(),
        "mlp_output_norm": wo.norm(dim=0).numpy(),
        "head_output_norm": np.array(head_out),
        "head_value_path_norm": np.array(head_value_path),
    }
    print("Layer", layer)
    print("  MLP input/output norm quantiles:",
          quantiles(wi.norm(dim=1)), quantiles(wo.norm(dim=0)))
    print("  MLP bias quantiles:", quantiles(bi))
    print("  Head output norms:", np.round(head_out, 3).tolist())
    print("  Head value-path norms:", np.round(head_value_path, 3).tolist())
    print("  Q/K row-norm ranges:",
          [round(float(t), 3) for t in (q.norm(dim=1).min(), q.norm(dim=1).max(),
                                       k.norm(dim=1).min(), k.norm(dim=1).max())])
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nSplit seed: 20261112. Selection indices: " + str(selection_q) + "\n")
    f.write("Static diagnostics: input/output norms and composed attention value/output weight norms. "
            "These are weight statistics, not an independently executed forward pass.\n")
