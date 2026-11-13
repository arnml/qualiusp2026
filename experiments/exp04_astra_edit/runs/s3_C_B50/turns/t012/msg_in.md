OUTPUT:
```
Testing heads: [(0, 2), (0, 1), (0, 5)]
[forward] pass used; 31 left
{'label': 'MLP_late75_heads_layer0', 'n': 120, 'original': 111, 'edited': 110, 'lost': [14, 23, 103], 'gained': [91, 124], 'margin_delta_quantiles': [-4.8766, -2.1251, -0.0433], 'score_correlation': 0.9633565823312615, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}
Testing heads: [(1, 2), (1, 9), (1, 8)]
[forward] pass used; 30 left
{'label': 'MLP_late75_heads_layer1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-4.3601, -1.965, -0.0812], 'score_correlation': 0.9659296438548238, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}
Testing heads: [(2, 11), (2, 0), (2, 2)]
[forward] pass used; 29 left
{'label': 'MLP_late75_heads_layer2', 'n': 120, 'original': 111, 'edited': 114, 'lost': [14], 'gained': [91, 124, 141, 192], 'margin_delta_quantiles': [-4.869, -2.0985, 0.0861], 'score_correlation': 0.9652184693371162, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}
Testing heads: [(3, 1), (3, 8), (3, 11)]
[forward] pass used; 28 left
{'label': 'MLP_late75_heads_layer3', 'n': 120, 'original': 111, 'edited': 112, 'lost': [14], 'gained': [91, 124], 'margin_delta_quantiles': [-5.2633, -2.7569, 0.2628], 'score_correlation': 0.9595261643158056, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}
Testing heads: [(4, 9), (4, 0), (4, 6)]
[forward] pass used; 27 left
{'label': 'MLP_late75_heads_layer4', 'n': 120, 'original': 111, 'edited': 110, 'lost': [14, 23], 'gained': [124], 'margin_delta_quantiles': [-4.5293, -2.1104, 0.0853], 'score_correlation': 0.9681088329275738, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}
Testing heads: [(5, 9), (5, 8), (5, 4)]
[forward] pass used; 26 left
{'label': 'MLP_late75_heads_layer5', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [124], 'margin_delta_quantiles': [-6.5473, -3.2988, -0.2689], 'score_correlation': 0.9581003499038825, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

```
Passes left: 26/50. Turn 11/200.