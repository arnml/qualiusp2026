```python
# Hypothesis: the frozen 4,096-neuron MLP edit preserves both-gold@3 on the untouched confirmation questions.
confirmation_result = forward(items=confirmation_items, model="both")
confirmation_summary = report_trial("confirmation_" + chosen_name, confirmation_result, confirmation_items)

confirmation_original = evaluate(confirmation_result["scores_original"], confirmation_items)
confirmation_edited = evaluate(confirmation_result["scores_edited"], confirmation_items)
for question_type in sorted({practice[q]["type"] for q in confirmation_q}):
    subset = [q for q in confirmation_q if practice[q]["type"] == question_type]
    print("Confirmation type", question_type, "n", len(subset),
          "original", sum(confirmation_original[q]["ok"] for q in subset),
          "edited", sum(confirmation_edited[q]["ok"] for q in subset))

final_cost = cost()
baseline_gflops = 39.2620032
reduction = 100 * (1 - final_cost["encoder_gflops_per_question"] / baseline_gflops)
quality_delta = 100 * (confirmation_summary["edited"] - confirmation_summary["original"]) / confirmation_summary["n"]
print("Final removal counts by layer:", {layer: len(indices) for layer, indices in chosen_config["neurons"].items()})
print("Final edits:", edited())
print("Encoder FLOPs reduction (%):", round(reduction, 3))
print("Confirmation quality delta (percentage points):", round(quality_delta, 3))

final_record = {
    "chosen": chosen_name,
    "discovery": experiments[chosen_name]["summary"],
    "confirmation": confirmation_summary,
    "cost": final_cost,
    "flops_reduction_percent": reduction,
    "confirmation_delta_percentage_points": quality_delta,
    "selection_used_confirmation": False,
}
(WORK / "results.json").write_text(json.dumps(final_record, indent=2), encoding="utf-8")
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\n## Final confirmation\n" + json.dumps(final_record, indent=2) +
            "\nAll ten forward passes used. Confirmation consisted of 80 previously unused questions "
            "evaluated through original and edited models on identical items. "
            "No final configuration changes were made using confirmation results. "
            "Finite practice samples do not establish a guaranteed held-out one-point bound. "
            "CPU timing is left to the physical-deletion evaluator.\n")
```
