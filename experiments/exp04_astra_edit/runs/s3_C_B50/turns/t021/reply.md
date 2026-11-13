```python
# Hypothesis: deleting the final MLP's neurons preserves confirmation ranking because the necessary late computation is in attention.
simplified_cfg = control_configs["remove_all_final_MLP"]
apply_config(**simplified_cfg)
out = forward(items=confirm_items, model="both", cls=True, mlp=True)
simplified_confirmation = report("FULL_FINAL_MLP_REMOVAL_CONFIRMATION", out, items=confirm_items)
if simplified_confirmation["edited"] >= simplified_confirmation["original"] and not simplified_confirmation["lost"]:
    final_cfg = simplified_cfg
    final_label = "early96_final_MLP_all"
    final_confirmation_scores = np.asarray(out["scores_edited"]).copy()
    print("Whole-final-MLP simplification accepted.")
else:
    apply_config(**configs["conservative_early96_late90"])
    final_cfg = configs["conservative_early96_late90"]
    final_label = "conservative_early96_late90"
    final_confirmation_scores = conservative_outputs[final_label][len(select_items):].copy()
    print("Whole-final-MLP simplification rejected; previously verified fallback restored.")
with NOTES.open("a") as f:
    f.write("\nFinal layer is necessary: removing layer 5 drops selection from 111/120 to 84/120, "
            "whereas removing its MLP neurons preserves all original selection successes. "
            "The evidence localizes essential late computation to attention and the retained residual/normalization path.\n")
    f.write("Final configuration after the confirmation gate: " + final_label + "\n")
print("Final edit:", edited())
```
