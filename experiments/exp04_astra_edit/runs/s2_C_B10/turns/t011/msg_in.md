OUTPUT:
```
Layer 0 scale 1.1915 surviving regular columns 880 protected high-energy columns 16
Layer 1 scale 1.1708 surviving regular columns 880 protected high-energy columns 16
Layer 2 scale 1.1374 surviving regular columns 880 protected high-energy columns 16
Layer 3 scale 1.2156 surviving regular columns 752 protected high-energy columns 16
Layer 4 scale 1.1502 surviving regular columns 624 protected high-energy columns 16
Layer 5 scale 1.25 surviving regular columns 240 protected high-energy columns 16
[forward] pass used; 4 left
{'name': 'mlp_B_scaled', 'original': 111, 'edited': 109, 'n': 120, 'losses': [29, 47, 86, 103], 'gains': [124, 139], 'margin_delta_quantiles': [-8.888, -5.144, -3.648, -1.471, 0.718], 'score_correlation': 0.93259, 'cost': {'encoder_gflops_per_question': 26.5611264, 'encoder_params': 6906368, 'removed': {'heads': 0, 'neurons': 4864, 'layers': []}}}

```
Passes left: 4/10. Turn 10/80.