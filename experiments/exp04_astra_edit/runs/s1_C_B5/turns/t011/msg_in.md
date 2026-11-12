OUTPUT:
```
[forward] pass used; 0 left
Selection: {'n': 100, 'original_successes': 92, 'edited_successes': 93, 'losses': [47], 'gains': [124, 139], 'margin_change_q10_median_q90': [-4.8861, -2.4556, 0.3293], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 71}, 'comparison': {'n': 22, 'original': 22, 'edited': 22}}}
Untouched confirmation: {'n': 100, 'original_successes': 97, 'edited_successes': 96, 'losses': [70], 'gains': [], 'margin_change_q10_median_q90': [-4.5597, -2.5129, -0.3693], 'by_type': {'bridge': {'n': 85, 'original': 82, 'edited': 81}, 'comparison': {'n': 15, 'original': 15, 'edited': 15}}}
All practice: {'n': 200, 'original_successes': 189, 'edited_successes': 189, 'losses': [47, 70], 'gains': [124, 139], 'margin_change_q10_median_q90': [-4.8265, -2.4764, 0.0495], 'by_type': {'bridge': {'n': 163, 'original': 152, 'edited': 152}, 'comparison': {'n': 37, 'original': 37, 'edited': 37}}}
Final cost: {'encoder_gflops_per_question': 29.9034624, 'encoder_params': 7890688, 'removed': {'heads': 0, 'neurons': 3584, 'layers': []}}
Encoder FLOP reduction (%): 23.836
Final edited state: {'track': 'C', 'heads_removed': [], 'neurons_removed': 3584, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 29.9034624, 'encoder_params': 7890688, 'removed': {'heads': 0, 'neurons': 3584, 'layers': []}}}

```
Passes left: 0/5. Turn 10/40. The forward-pass budget is used up. Code still runs (forward() is refused). Make sure /work/edited holds your final edit, then reply WITHOUT a code block with your final explanation (mechanism, changes, evidence).