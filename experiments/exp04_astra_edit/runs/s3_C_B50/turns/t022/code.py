# Hypothesis: the final 2,016-neuron removal preserves every original practice success while reducing encoder FLOPs by about 13.4%.
import json

apply_config(**final_cfg)
final_out = forward(items=all_items, model="both")
original_scores = np.asarray(final_out["scores_original"]).reshape(-1)
edited_scores = np.asarray(final_out["scores_edited"]).reshape(-1)
original_metrics = question_metrics(original_scores, all_items)
edited_metrics = question_metrics(edited_scores, all_items)

final_summary = {}
for split, qs in [("selection", select_q), ("confirmation", confirm_q), ("all_practice", select_q + confirm_q)]:
    old_ok = sum(original_metrics[q][0] for q in qs)
    new_ok = sum(edited_metrics[q][0] for q in qs)
    lost = [q for q in qs if original_metrics[q][0] and not edited_metrics[q][0]]
    gained = [q for q in qs if not original_metrics[q][0] and edited_metrics[q][0]]
    finite_deltas = [
        edited_metrics[q][1] - original_metrics[q][1] for q in qs
        if np.isfinite(edited_metrics[q][1]) and np.isfinite(original_metrics[q][1])
    ]
    row = {
        "questions": len(qs), "original_successes": old_ok, "edited_successes": new_ok,
        "original_percent": 100 * old_ok / len(qs), "edited_percent": 100 * new_ok / len(qs),
        "paired_losses": lost, "paired_gains": gained,
        "finite_margin_delta_quantiles": np.quantile(finite_deltas, [.1, .5, .9]).tolist(),
    }
    final_summary[split] = row
    print(split, row)

final_cost = cost()
final_summary["cost"] = final_cost
final_summary["flops_reduction_percent"] = 100 * (1 - final_cost["encoder_gflops_per_question"] / 39.2620032)
final_summary["parameter_reduction_percent"] = 100 * (1 - final_cost["encoder_params"] / 10646784)
final_summary["validation_caveat"] = (
    "The 120/80 split was fixed before diagnosis. Component salience and initial candidate choices "
    "used the 120-question selection split. Confirmation rejected aggressive cuts and gated rollback; "
    "it is therefore not an untouched final estimate. No held-out 1,000-question results are available."
)
manifest = {
    "label": final_label,
    "neurons_removed": {str(l): [int(i) for i in ids] for l, ids in final_cfg["neurons"].items()},
    "heads_removed": final_cfg["heads"], "layers_removed": final_cfg["layers"], "scalings": [],
    "selection_questions": select_q, "confirmation_questions": confirm_q,
    "summary": final_summary,
}
(WORK / "pruning_manifest.json").write_text(json.dumps(manifest, indent=2))
with NOTES.open("a") as f:
    f.write("\n## Final submission\n")
    f.write("Removed 96 lowest weight-salience MLP neurons in each of layers 0–4, "
            "and all 1536 MLP neurons in layer 5. All attention heads, layer boundaries, "
            "biases, normalization parameters, embeddings, pooler, and classifier retained. No scaling.\n")
    f.write(json.dumps(final_summary, indent=2) + "\n")
    f.write("Mechanistic evidence: final MLP removal preserves ranking; whole final-layer removal "
            "fails badly. Early removals target low incoming-row-plus-bias norm times outgoing-column norm. "
            "Removing equally many high-strength early channels causes paired regressions. "
            "No gradients, fitting, or custom network execution used.\n")

print("Final cost:", final_cost)
print("FLOPs reduction:", round(final_summary["flops_reduction_percent"], 3), "%")
print("Final state:", edited())
print("Notes:", str(NOTES))
print("Manifest:", str(WORK / "pruning_manifest.json"))
