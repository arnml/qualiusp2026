```python
# Hypothesis: keeping at least 87.5% of each earlier MLP will protect paragraph features, while the final MLP still offers substantial redundancy.
conservative_results = {}
conservative_outputs = {}
for early_count in [192, 96, 0]:
    label = f"conservative_early{early_count}_late90"
    counts = [early_count] * 5 + [1382]
    configs[label] = {
        "neurons": {l: (neuron_ranks[l] if l == 5 else weight_ranks[l])[:counts[l]].tolist()
                    for l in range(6)},
        "heads": [], "layers": [],
    }
    apply_config(**configs[label])
    out = forward(items=all_items, model="both")
    ns = len(select_items)
    sm = question_metrics(np.asarray(out["scores_edited"])[:ns], select_items)
    vm = question_metrics(np.asarray(out["scores_edited"])[ns:], confirm_items)
    vb = question_metrics(np.asarray(out["scores_original"])[ns:], confirm_items)
    row = {
        "label": label,
        "selection": [sum(x[0] for x in base_m.values()), sum(x[0] for x in sm.values())],
        "selection_lost": sum(base_m[q][0] and not sm[q][0] for q in sm),
        "confirmation": [sum(x[0] for x in vb.values()), sum(x[0] for x in vm.values())],
        "confirmation_lost": sum(vb[q][0] and not vm[q][0] for q in vm),
        "confirmation_gained": sum(not vb[q][0] and vm[q][0] for q in vm),
        "score_correlation": float(np.corrcoef(out["scores_original"], out["scores_edited"])[0, 1]),
        "cost": cost(),
    }
    conservative_results[label] = row
    conservative_outputs[label] = np.asarray(out["scores_edited"]).copy()
    print(row)
    with NOTES.open("a") as f:
        f.write("\n" + repr(row) + "\n")
reset_all()
```
