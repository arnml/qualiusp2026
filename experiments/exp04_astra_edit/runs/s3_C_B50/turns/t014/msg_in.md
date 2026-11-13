OUTPUT:
```
[forward] pass used; 22 left
{'label': 'MLP_balanced2', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 171], 'margin_delta_quantiles': [-4.2518, -1.894, 0.4668], 'score_correlation': 0.96600069544218, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}
[forward] pass used; 21 left
{'label': 'MLP_balanced3', 'n': 120, 'original': 111, 'edited': 112, 'lost': [66], 'gained': [91, 171], 'margin_delta_quantiles': [-4.6678, -1.9652, 0.4404], 'score_correlation': 0.9679198305604829, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}
[forward] pass used; 20 left
{'label': 'balanced_heads1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-4.1718, -1.7162, 0.3666], 'score_correlation': 0.9671287893864661, 'cost': {'encoder_gflops_per_question': 29.3590272, 'encoder_params': 7762938, 'removed': {'heads': 3, 'neurons': 3558, 'layers': []}}}

```
Passes left: 20/50. Turn 13/200.