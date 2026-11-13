OUTPUT:
```
{'encoder_gflops_per_question': 28.795008, 'encoder_params': 7724876, 'removed': {'heads': 3, 'neurons': 1300, 'layers': [1]}}
[forward] pass used; 3 left
{'scores_edited': (1992,), 'cls_edited': (1992, 6, 384), 'mlp_edited': (1992, 5, 1536), 'passes_left': 3}
A: orig 0.910 edited 0.840  lost 9 gained 2  mean within-q spearman 0.761
B: orig 0.980 edited 0.810  lost 17 gained 0  mean within-q spearman 0.754

```
Passes left: 3/5. Turn 13/40.