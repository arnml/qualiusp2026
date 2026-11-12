Removing layer 1 caused 14 paired failures, so I restored it. The first MLP edit looks more promising: it removes 3,456 neurons, reduces encoder cost by 23%, and scores 112/120 versus 111/120 on discovery questions. It does shrink ranking margins, so I’ll check that pressure as I test further removals.


```python
# Hypothesis: heads with weak value-to-output maps and redundant attention patterns are the safest next removals.
head_diagnostics = {}
for layer in range(6):
    prefix = f"bert.encoder.layer.{layer}."
    v = load_original(prefix + "attention.self.value.weight").numpy().reshape(12, 32, 384)
    o = load_original(prefix + "attention.output.dense.weight").numpy().reshape(384, 12, 32)
    effective_norm = []
    readout_norm = []
    for head in range(12):
        # These are static weight-map norms, not model activations or a reconstructed forward pass.
        weight_map = o[:, head, :] @ v[head]
        effective_norm.append(float(np.linalg.norm(weight_map)))
        readout_norm.append(float(np.linalg.norm(weight_direction @ weight_map)))
    cls_attention = np.asarray(baseline["attn_original"][:, layer, :, 0, :], dtype=np.float32)
    profiles = cls_attention.transpose(1, 0, 2).reshape(12, -1)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    similarities = profiles @ profiles.T
    np.fill_diagonal(similarities, -1)
    neighbor = similarities.argmax(axis=1)
    head_diagnostics[layer] = {
        "map_norm": effective_norm, "readout_norm": readout_norm,
        "attention_neighbor": neighbor.tolist(), "attention_similarity": similarities.max(axis=1).tolist(),
    }
    print("Layer", layer, "head: map norm / static readout norm / most similar CLS-attention head, cosine")
    print(" ".join(f"{h}:{effective_norm[h]:.2f}/{readout_norm[h]:.3f}/{neighbor[h]},{similarities[h, neighbor[h]]:.2f}"
                   for h in range(12)))
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nMLP A retained as an eligible discovery candidate: 112/120 versus 111/120, "
            "one loss and two gains, encoder 30.237696 GFLOPs/question. "
            "Margins shrank, so score correlation and paired transitions remain selection checks. "
            "Additional head diagnostics use static O_h V_h Frobenius/readout norms and "
            "cosine similarity of observed CLS-attention profiles, with no network reconstruction.\n")
```
