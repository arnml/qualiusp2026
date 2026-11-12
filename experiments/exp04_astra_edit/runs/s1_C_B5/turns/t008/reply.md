The attention sample suggests preserving the later heads that focus on question terms; one also carries substantial paragraph-to-question attention. I’ll test eight earlier or weaker heads while keeping the MLP edit fixed, so this pass isolates the added effect of head removal.

```python
# Hypothesis: eight weak or CLS-focused heads add little task information beyond the remaining heads.
tested_heads = [
    (0, 2),
    (1, 6), (1, 7), (1, 9),
    (2, 1), (2, 11),
    (3, 1),
    (4, 9),
]
plans["B_mlp_heads"] = make_plan([384, 512, 512, 512, 640, 1024], heads=tested_heads)
apply_plan(plans["B_mlp_heads"])
print("Candidate B cost:", cost())
results["B_mlp_heads"] = forward(items=baseline_items, model="both", cls=True)
report_b = compare_result(baseline_items, results["B_mlp_heads"])
print("Candidate B selection:", report_b)

metrics_a = question_metrics(baseline_items, results["A_mlp"]["scores_edited"])
metrics_b = question_metrics(baseline_items, results["B_mlp_heads"]["scores_edited"])
print("Incremental head-removal losses:",
      [q for q in selection_q if metrics_a[q]["success"] and not metrics_b[q]["success"]])
print("Incremental head-removal gains:",
      [q for q in selection_q if not metrics_a[q]["success"] and metrics_b[q]["success"]])
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nCandidate B adds heads " + str(tested_heads) + " to candidate A.\n")
    f.write("Evidence: layer 0 head 2 attends exclusively to CLS from CLS; layer 1 candidates predominantly "
            "attend to CLS there and have low cross-segment attention; layer 2/3 candidates combine weak "
            "weight paths with little cross-segment attention; layer 4 head 9 has the weakest value/output "
            "path in its layer. Later lexical question-attending heads were retained. These are hypotheses, "
            "not claims of zero contribution.\n")
    f.write("Pass 3 paired selection result: " + str(report_b) + "\nCost: " + str(cost()) + "\n")
```
