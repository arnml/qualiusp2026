The 4,096-neuron edit matches the lighter edit’s discovery result: 112/120 versus 111/120, with one paired loss and two gains. It cuts encoder FLOPs by 27.2%. I’ll make one final discovery test of two weak late heads, then freeze the selection before evaluating the reserved 80 questions.


```python
# Hypothesis: removing only the weakest layer-4 head and lowest-readout layer-5 head may preserve the successful MLP edit.
configs["mlp_C_heads2"] = copy.deepcopy(configs["mlp_C"])
configs["mlp_C_heads2"]["heads"] = [(4, 9), (5, 11)]
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nMLP D: 110/120, three losses and two gains, 27.2295936 GFLOPs/question. "
            "Although its net decline is 0.83 points, its additional paired losses and weaker "
            "score correlation make C the more robust candidate. "
            "Final discovery experiment isolates L4H9 (weakest value/output map and readout norm "
            "in that layer) and L5H11 (weakest static readout norm), rather than the rejected eight-head set. "
            "Selection rule before confirmation: require at least the original 111/120 and at most "
            "one paired loss; among eligible candidates select lowest FLOPs, excluding whole-layer ablations. "
            "If the two-head version fails this rule, choose C. Confirmation labels/scores remain unused.\n")
apply_config(configs["mlp_C_heads2"])
last_discovery_trial = forward(items=discovery_items, model="both")
report_trial("mlp_C_heads2", last_discovery_trial)

eligible = [name for name in ["mlp_A", "mlp_C", "mlp_C_heads2"]
            if experiments[name]["summary"]["edited"] >= 111
            and len(experiments[name]["summary"]["losses"]) <= 1]
chosen_name = min(eligible, key=lambda name: experiments[name]["summary"]["cost"]["encoder_gflops_per_question"])
chosen_config = copy.deepcopy(configs[chosen_name])
apply_config(chosen_config)
print("FROZEN SELECTION:", chosen_name)
print("Frozen cost:", cost())
(WORK / "selected_edit.json").write_text(json.dumps(chosen_config, indent=2), encoding="utf-8")
with NOTES.open("a", encoding="utf-8") as f:
    f.write(f"\nFrozen selection before confirmation: {chosen_name}. "
            "No changes will be selected using the reserved confirmation results.\n")
```
