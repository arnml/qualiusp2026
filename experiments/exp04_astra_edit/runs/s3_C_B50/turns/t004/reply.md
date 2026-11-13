Layers 1 and 2 change the CLS representation relatively little; layers 4 and 5 contribute much more along the classifier’s readout direction. I’ll test whether those small early-layer changes are nevertheless necessary for ranking, alongside the low-activation MLP channels.


```python
# Hypothesis: layers 1–2 may be redundant, but only paired ablations can distinguish small necessary updates from dispensable ones.
tokens = np.asarray(diag["attn_tokens_original"])
attn = diag["attn_original"]
head_diagnostics = {}
for l in range(6):
    rows = []
    for h in range(12):
        sinks, cls_sep, entropy = [], [], []
        for i in range(len(tokens)):
            valid = tokens[i] != "[PAD]"
            sink = np.isin(tokens[i], ["[SEP]", "[CLS]", ".", ",", "?", ";", ":"])
            a = attn[i, l, h]
            sinks.append(float(a[valid][:, sink].sum(axis=1).mean()))
            cls_sep.append(float(a[0, tokens[i] == "[SEP]"].sum()))
            p = a[0, valid]
            entropy.append(float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(valid.sum())))
        rows.append((h, round(np.mean(sinks), 3), round(np.mean(cls_sep), 3), round(np.mean(entropy), 3)))
        head_diagnostics[(l, h)] = (float(np.mean(sinks)), float(np.mean(cls_sep)), float(np.mean(entropy)))
    print("Layer", l, "(head, sink mass, CLS→SEP, CLS entropy):", rows)
    print("  Near-zero CLS channels:", int((neuron_importance[l] < 1e-5).sum()))

layer_results = {}
for l in [1, 2]:
    remove_layer(l)
    out = forward(items=select_items, model="both")
    layer_results[l] = report(f"remove layer {l}", out)
    restore_layer(l)

with NOTES.open("a") as f:
    f.write("\nDiagnostics: CLS residual cosine is 0.980/0.978 in layers 1/2; "
            "late layers show larger readout-direction changes. MLP importance is "
            "RMS(original CLS GELU activation) times outgoing-column L2 norm. "
            "CLS inactivity alone is not evidence about other tokens; causal tests are required.\n")
```
