OUTPUT:
```
C:\code\qualiusp2026\experiments\exp04_astra_edit\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:3037: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
C:\code\qualiusp2026\experiments\exp04_astra_edit\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:3038: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
CLS stage 0 correlation with original score: nan
CLS stage 1 correlation with original score: 0.1615
CLS stage 2 correlation with original score: -0.0836
CLS stage 3 correlation with original score: -0.1872
CLS stage 4 correlation with original score: 0.2578
CLS stage 5 correlation with original score: 0.743
CLS stage 6 correlation with original score: 0.9887
[forward] pass used; 2 left
{'label': 'MECHANISTIC_CONTROL_remove_layer5', 'n': 120, 'original': 111, 'edited': 84, 'lost': [10, 13, 14, 16, 18, 22, 23, 34, 50, 57, 60, 61, 66, 74, 90, 100, 101, 102, 103, 111, 115, 116, 119, 156, 169, 179, 184, 187, 189, 196], 'gained': [91, 124, 192], 'margin_delta_quantiles': [-11.7123, -6.6517, -1.4285], 'score_correlation': 0.7290325347116708, 'cost': {'encoder_gflops_per_question': 31.5485184, 'encoder_params': 8503200, 'removed': {'heads': 0, 'neurons': 480, 'layers': [5]}}}

```
Passes left: 2/50. Turn 20/200.