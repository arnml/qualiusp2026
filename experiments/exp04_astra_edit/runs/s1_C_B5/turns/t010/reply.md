The larger MLP edit also fell to 88% on the selection half. The chosen edit is therefore the earlier **3,584-neuron removal**, with every head and layer retained. I’ll use the final pass to check it against the original on all 200 questions, reporting the untouched confirmation half separately.

```python
# Hypothesis: the selection-validated 3,584-neuron edit preserves both-gold@3 on untouched confirmation questions.
assert preferred_candidate == "A_mlp"
apply_plan(plans[preferred_candidate])
results["final_full"] = forward(items=all_items, model="both")
final_selection_report = compare_result(all_items, results["final_full"], selection_q)
final_confirmation_report = compare_result(all_items, results["final_full"], confirmation_q)
final_all_report = compare_result(all_items, results["final_full"])
final_cost = cost()
savings_percent = 100 * (1 - final_cost["encoder_gflops_per_question"] / 39.2620032)

print("Selection:", final_selection_report)
print("Untouched confirmation:", final_confirmation_report)
print("All practice:", final_all_report)
print("Final cost:", final_cost)
print("Encoder FLOP reduction (%):", round(savings_percent, 3))
print("Final edited state:", edited())

import json
Path("/work/removal_plan.json").write_text(
    json.dumps(plans[preferred_candidate], indent=2), encoding="utf-8"
)
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nCandidate C rejected on selection alone: 88/100 and score correlation 0.9369. "
            "Its confirmation outcomes were not inspected or used.\n")
    f.write("Final chosen candidate A restored before pass 5: "
            "3584 neurons, counts [384,512,512,512,640,1024]; all heads/layers retained; no rescalings.\n")
    f.write("Pass 5 selection: " + str(final_selection_report) + "\n")
    f.write("Pass 5 previously untouched confirmation: " + str(final_confirmation_report) + "\n")
    f.write("Pass 5 all practice: " + str(final_all_report) + "\n")
    f.write("Final cost: " + str(final_cost) + "\nFLOP reduction: " + str(savings_percent) + "%.\n")
    f.write("Exact neuron indices saved in /work/removal_plan.json. CPU timing was not measured here.\n")
```
