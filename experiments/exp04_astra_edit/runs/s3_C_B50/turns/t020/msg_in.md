OUTPUT:
```
Layer 0 removed 96 remaining 1440 removed share of summed CLS channel energy proxy: 5.4e-05
Layer 1 removed 96 remaining 1440 removed share of summed CLS channel energy proxy: 0.000508
Layer 2 removed 96 remaining 1440 removed share of summed CLS channel energy proxy: 0.000663
Layer 3 removed 96 remaining 1440 removed share of summed CLS channel energy proxy: 0.005784
Layer 4 removed 96 remaining 1440 removed share of summed CLS channel energy proxy: 0.026313
Layer 5 removed 1382 remaining 154 removed share of summed CLS channel energy proxy: 0.113269
[forward] pass used; 4 left
{'label': 'MECHANISTIC_CONTROL_remove_all_final_MLP', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [141], 'margin_delta_quantiles': [-0.8406, 0.2901, 1.2966], 'score_correlation': 0.9932959339563128, 'cost': {'encoder_gflops_per_question': 33.997824, 'encoder_params': 9096480, 'removed': {'heads': 0, 'neurons': 2016, 'layers': []}}}
[forward] pass used; 3 left
{'label': 'MECHANISTIC_CONTROL_remove_high_strength_early_channels', 'n': 120, 'original': 111, 'edited': 112, 'lost': [13, 111], 'gained': [89, 91, 141], 'margin_delta_quantiles': [-2.1247, -0.4813, 1.5411], 'score_correlation': 0.9787664787362031, 'cost': {'encoder_gflops_per_question': 34.3999488, 'encoder_params': 9214906, 'removed': {'heads': 0, 'neurons': 1862, 'layers': []}}}
Frozen final model restored: {'track': 'C', 'heads_removed': [], 'neurons_removed': 1862, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 34.3999488, 'encoder_params': 9214906, 'removed': {'heads': 0, 'neurons': 1862, 'layers': []}}}

```
Passes left: 3/50. Turn 19/200.