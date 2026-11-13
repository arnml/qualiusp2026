```python
# Hypothesis: weak residual branches and low-use MLP channels can be identified from weights and selection-only activations.
def wn(layer, suffix):
    return f"bert.encoder.layer.{layer}.{suffix}"

def arr(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def describe(x):
    if isinstance(x, dict):
        return {str(k): describe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return {"length": len(x), "first": describe(x[0]) if len(x) else None}
    return {"shape": tuple(x.shape)} if hasattr(x, "shape") else type(x).__name__

for l in range(6):
    v = arr(load_original(wn(l, "attention.self.value.weight"))).reshape(12, 32, 384)
    o = arr(load_original(wn(l, "attention.output.dense.weight"))).reshape(384, 12, 32)
    head_strength = np.linalg.norm(v, axis=(1, 2)) * np.linalg.norm(o, axis=(0, 2))
    wi = arr(load_original(wn(l, "intermediate.dense.weight")))
    wo = arr(load_original(wn(l, "output.dense.weight")))
    neuron_strength = np.linalg.norm(wi, axis=1) * np.linalg.norm(wo, axis=0)
    print("Layer", l, "head strength:", np.round(head_strength, 2).tolist(),
          "MLP strength quantiles:", np.round(np.quantile(neuron_strength, [0, .1, .5, .9, 1]), 3).tolist())

diag = forward(items=select_items, model="original", cls=True, mlp=True, attn_n=8)
print("Diagnostic structure:", describe(diag))
with NOTES.open("a") as f:
    f.write("\nPass 1: original-model CLS, MLP, and eight attention diagnostics on selection paragraphs only.\n")
```
