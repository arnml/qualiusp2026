OUTPUT:
```
[forward] pass used; 1 left
{'name': 'mlp_C_heads2', 'original': 111, 'edited': 110, 'n': 120, 'losses': [23, 47], 'gains': [124], 'margin_delta_quantiles': [-7.15, -4.431, -2.902, -0.841, 0.79], 'score_correlation': 0.95667, 'cost': {'encoder_gflops_per_question': 28.1583104, 'encoder_params': 7398464, 'removed': {'heads': 2, 'neurons': 4096, 'layers': []}}}
FROZEN SELECTION: mlp_C
Frozen cost: {'encoder_gflops_per_question': 28.566528, 'encoder_params': 7496960, 'removed': {'heads': 0, 'neurons': 4096, 'layers': []}}

```
Passes left: 1/10. Turn 13/80.