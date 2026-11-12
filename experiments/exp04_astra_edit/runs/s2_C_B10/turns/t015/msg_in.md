OUTPUT:
```
[forward] pass used; 0 left
{'name': 'confirmation_mlp_C', 'original': 78, 'edited': 77, 'n': 80, 'losses': [70], 'gains': [], 'margin_delta_quantiles': [-6.128, -4.018, -2.315, -1.457, -0.088], 'score_correlation': 0.96146, 'cost': {'encoder_gflops_per_question': 28.566528, 'encoder_params': 7496960, 'removed': {'heads': 0, 'neurons': 4096, 'layers': []}}}
Confirmation type bridge n 69 original 67 edited 66
Confirmation type comparison n 11 original 11 edited 11
Final removal counts by layer: {0: 512, 1: 512, 2: 512, 3: 640, 4: 640, 5: 1280}
Final edits: {'track': 'C', 'heads_removed': [], 'neurons_removed': 4096, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 28.566528, 'encoder_params': 7496960, 'removed': {'heads': 0, 'neurons': 4096, 'layers': []}}}
Encoder FLOPs reduction (%): 27.241
Confirmation quality delta (percentage points): -1.25

```
Passes left: 0/10. Turn 14/80. The forward-pass budget is used up. Code still runs (forward() is refused). Make sure /work/edited holds your final edit, then reply WITHOUT a code block with your final explanation (mechanism, changes, evidence).