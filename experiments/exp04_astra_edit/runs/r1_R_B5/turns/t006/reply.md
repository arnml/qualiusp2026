```python
# Hypothesis: output-weight geometry distinguishes neurons that promote lexical distractors from neurons that suppress them.
diagnostic_layers = []
score_center = centered(dscores, groups_d)
for layer in range(2, 6):
    acts = baseline["mlp_original"][drows, layer, :]
    hidden = baseline["cls_original"][drows, layer + 1, :]
    ch = centered(hidden, groups_d)
    score_axis = (ch * score_center[:, None]).mean(axis=0)
    score_axis /= max(np.linalg.norm(score_axis), 1e-12)
    w = load_original(f"bert.encoder.layer.{layer}.output.dense.weight").numpy()
    gamma = load_original(f"bert.encoder.layer.{layer}.output.LayerNorm.weight").numpy()
    alignment = (gamma[:, None] * w).T @ score_axis
    contrast_raw = acts[competing_distractor].mean(axis=0) - acts[weak_gold].mean(axis=0)
    st = neuron_stats[layer]
    eligible = (st["overlap"] > 0.15) & (st["score"] > 0.15) & (alignment > 0) & (contrast_raw > 0)
    priority = np.where(eligible, alignment * contrast_raw * np.maximum(st["overlap"] - st["gold"], 0), 0)
    selected = np.argsort(-priority)[:10]
    diagnostic_layers.append({"layer": layer, "alignment": alignment, "priority": priority,
                              "eligible": eligible, "score_axis": score_axis})
    print("Layer", layer, "eligible:", int(eligible.sum()))
    print("(neuron, priority, alignment, distractor contrast, overlap r, gold r):",
          [(int(i), round(float(priority[i]), 4), round(float(alignment[i]), 4),
            round(float(contrast_raw[i]), 3), round(float(st["overlap"][i]), 3),
            round(float(st["gold"][i]), 3)) for i in selected])
    print("Top32 priority share:", round(float(np.sort(priority)[-32:].sum() /
                                              max(priority.sum(), 1e-12)), 3))
print("Current edits:", edited())
```
