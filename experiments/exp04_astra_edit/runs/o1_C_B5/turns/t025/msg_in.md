OUTPUT:
```
{'encoder_gflops_per_question': 35.4688, 'encoder_params': 9573132, 'removed': {'heads': 4, 'neurons': 1140, 'layers': []}}
[forward] pass used; 0 left
A: orig 0.910 edited 0.900  lost 2 gained 1  mean within-q spearman 0.981
B: orig 0.980 edited 0.980  lost 0 gained 0  mean within-q spearman 0.985
score diff mean -0.14 |diff| 0.32 within-q std 0.398
P5 flips: [(13, 1, 0, True), (119, 1, 0, True), (141, 0, 1, True)]
CLS cos per state P5: [1.0, 1.0, 0.99999, 0.99993, 0.99966, 0.9986, 0.99541]

```
Passes left: 0/5. Turn 24/40. The forward-pass budget is used up. Code still runs (forward() is refused). Make sure your current removals are your final edit, then reply WITHOUT a code block with your final explanation (mechanism, changes, evidence).