The discovery set confirms a substantial overlap bias: scores correlate 0.60 with question-word coverage. Several failures favor paragraphs that repeat the named entity over gold paragraphs containing the requested location, nationality, or team. The strongest candidate signals appear in layers 3–4, but activation correlation alone does not tell us whether a neuron promotes or suppresses those matches. I’ll check the directions of its output weights before intervening.

```python
# Hypothesis: a small subset of late MLP outputs actively promotes lexical distractors rather than merely responding to them.
diagnostic_layers = []
score_center = centered(dscores, groups_d)
overlap_center = centered(doverlap, groups_d)
gold_center = centered(dgold, groups_d)
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
    effect = alignment * contrast_raw
    st = neuron_stats[layer]
    # The screen requires both a lexical association and an output direction that promotes score.
    eligible = (st["overlap"] > 0.15) & (st["score"] > 0.15) & (alignment > 0) & (contrast_raw > 0)
    priority = np.where(eligible, effect * np.maximum(st["overlap"] - st["gold"], 0), 0)
    selected = np.argsort(-priority)[:12]
    diagnostic_layers.append({"layer": layer, "alignment": alignment, "effect": effect,
                              "priority": priority, "eligible": eligible, "score_axis": score_axis})
    print("\nLayer", layer, "eligible:", int(eligible.sum()))
    print("(neuron, priority, output alignment, distractor-gold activation, overlap r, gold r)")
    print([(int(i), round(float(priority[i]), 4), round(float(alignment[i]), 4),
            round(float(contrast_raw[i]), 3), round(float(st["overlap"][i]), 3),
            round(float(st["gold"][i]), 3)) for i in selected])
    print("Top 32 share of positive priority:", round(float(np.sort(priority)[-32:].sum() /
                                                           max(priority.sum(), 1e-12)), 3))

print("\nFailure rates by type:")
for typ in sorted(set(practice[int(q)]["type"] for q in discovery_ids)):
    mask = np.array([practice[int(q)]["type"] == typ for q in discovery_ids])
    print(typ, "n =", int(mask.sum()), "both@3 =", round(float(base_d["success"][mask].mean()), 3))
print("Current edits:", edited())
```
