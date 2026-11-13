The candidate failed confirmation: it lost four of the original model’s 78 successes. I’m rejecting it and checking the more conservative configurations already tested on the selection split. I won’t tune individual components to those four questions.


```python
# Hypothesis: rolling back to pre-evaluated, less extensive MLP pruning will recover confirmation quality.
all_items = select_items + confirm_items
fallback_results = {}
fallback_outputs = {}
for label in ["MLP_balanced", "MLP_late90", "MLP_weight_25"]:
    apply_config(**configs[label])
    out = forward(items=all_items, model="both")
    ns = len(select_items)
    train_part = {k: np.asarray(out[k])[:ns] for k in ["scores_original", "scores_edited"]}
    confirm_part = {k: np.asarray(out[k])[ns:] for k in ["scores_original", "scores_edited"]}
    report(label + "_selection_recheck", train_part)
    fallback_results[label] = report(label + "_confirmation", confirm_part, items=confirm_items)
    fallback_outputs[label] = np.asarray(out["scores_edited"]).copy()
reset_all()
with NOTES.open("a") as f:
    f.write("\nAggressive candidate rejected: confirmation 74/80 vs 78/80. "
            "Checking predeclared rollback configurations; confirmation is now a rollback gate, "
            "not an untouched final estimate. No per-example component tuning is performed.\n")
```
