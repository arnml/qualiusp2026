At 50% attenuation, discovery token retention fell from 950 to 913 tokens per question, while both-gold@3 stayed at 55%. The ranking-margin change was small. I’ll test a stronger subtraction of the same 96 outputs to see whether the effect grows with strength.

```python
# Hypothesis: subtracting twice the selected lexical-feature contribution will produce a larger, consistent reduction in distractor retention.
install_attenuation(2.0)
trial_double = forward(items=discovery_items, model="both", cls=True, mlp=True)
double_metrics, double_report = paired_report(trial_double["scores_edited"])
trials[2.0] = {"output": trial_double, "metrics": double_metrics, "report": double_report}
print("Strength 2.0:", double_report)
print("Paired uncertainty:",
      paired_bootstrap(trial_double["scores_original"], trial_double["scores_edited"],
                       groups_d, dgold, dlength))
for strength in [0.5, 2.0]:
    change = trials[strength]["output"]["scores_edited"] - dscores
    lower_gold = weak_gold[np.argsort(dscores[weak_gold])[:20]]
    print("Strength", strength, "score changes:",
          {"lowest_20_gold": round(float(change[lower_gold].mean()), 4),
           "all_weak_gold": round(float(change[weak_gold].mean()), 4),
           "competing_distractors": round(float(change[competing_distractor].mean()), 4),
           "all_distractors": round(float(change[~dgold].mean()), 4),
           "change_vs_overlap_r": round(float(correlations(centered(change, groups_d),
                                                          centered(doverlap, groups_d))), 4)})
with NOTES.open("a") as f:
    f.write("Strength 2.0: " + repr(double_report) + "\n")
```
