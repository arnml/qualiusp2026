# Hypothesis: the candidate selected on 120 questions will preserve both-gold@3 on the untouched 80-question confirmation split.
candidate_label = "balanced2_head4_9"
apply_config(**configs[candidate_label])
with NOTES.open("a") as f:
    f.write("\nCandidate frozen before confirmation: balanced2_head4_9. "
            "Neuron removals by layer: [256, 768, 256, 512, 768, 1382]; head (4,9); "
            "no layer removals or scalings. Selection: 114/120 vs 111/120, no original success lost. "
            "Confirmation has not been inspected. If confirmation rejects the candidate, "
            "rollback options are previously evaluated MLP_balanced, MLP_late90, and MLP_weight_25; "
            "no individual components will be tuned to confirmation examples.\n")
confirm_out = forward(items=confirm_items, model="both", cls=True, mlp=True, attn_n=8)
confirm_row = report("FROZEN_CANDIDATE_CONFIRMATION", confirm_out, items=confirm_items)
confirm_base = np.asarray(confirm_out["scores_original"]).copy()
confirm_candidate = np.asarray(confirm_out["scores_edited"]).copy()
print("Candidate:", edited())
