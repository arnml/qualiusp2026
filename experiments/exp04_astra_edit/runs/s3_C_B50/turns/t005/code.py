# Hypothesis: separator-focused attention and small residual updates identify candidates, but their necessity must be tested causally.
head_diagnostics = {}
for l in range(6):
    rows = []
    for h in range(12):
        sinks, cls_sep, entropy = [], [], []
        for i in range(len(tokens)):
            valid = tokens[i] != "[PAD]"
            sink = np.isin(tokens[i], ["[SEP]", "[CLS]", ".", ",", "?", ";", ":"])
            a = np.asarray(attn[i, l, h], dtype=np.float32)
            sinks.append(float(a[valid][:, sink].sum(axis=1).mean()))
            cls_sep.append(float(a[0, tokens[i] == "[SEP]"].sum()))
            p = a[0, valid]
            entropy.append(float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(valid.sum())))
        rows.append((h, round(float(np.mean(sinks)), 3), round(float(np.mean(cls_sep)), 3),
                     round(float(np.mean(entropy)), 3)))
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
    f.write("\nDiagnostics: CLS residual cosine is 0.980/0.978 in layers 1/2; late layers "
            "show larger readout-direction changes. MLP importance is RMS(original CLS "
            "GELU activation) times outgoing-column L2 norm. CLS inactivity alone does not "
            "establish inactivity at other tokens; causal tests are required.\n")
