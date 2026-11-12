OUTPUT:
```
Candidate B cost: {'encoder_gflops_per_question': 28.270592, 'encoder_params': 7496704, 'removed': {'heads': 8, 'neurons': 3584, 'layers': []}}
[forward] pass used; 2 left
Candidate B selection: {'n': 100, 'original_successes': 92, 'edited_successes': 88, 'losses': [14, 29, 47, 86, 103, 156], 'gains': [124, 192], 'margin_change_q10_median_q90': [-7.5442, -2.9288, -0.0076], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 67}, 'comparison': {'n': 22, 'original': 22, 'edited': 21}}}
Incremental head-removal losses: [14, 29, 86, 103, 139, 156]
Incremental head-removal gains: [192]

```
Passes left: 2/5. Turn 8/40.