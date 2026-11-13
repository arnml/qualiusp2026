OUTPUT:
```
{'bert.encoder.layer.4.output.dense.weight': 33408}
[forward] pass used; 0 left
saved; passes_left: 0
A orig {'tok85': 829.9, 'tok90': 945.51, 'tok95': 1026.5, 'b3': 0.58, 'AUC': 0.72}
   a=1  {'tok85': 819.39, 'tok90': 932.06, 'tok95': 1005.95, 'b3': 0.58, 'AUC': 0.724} | tok90 Δ (-14.9, 19.3) | tok85 Δ (-15.9, 22.3)
   a=3  {'tok85': 791.26, 'tok90': 986.63, 'tok95': 1056.53, 'b3': 0.57, 'AUC': 0.716} | tok90 Δ (10.0, 62.9) | tok85 Δ (1.6, 83.5)
B orig {'tok85': 911.7, 'tok90': 963.06, 'tok95': 1044.46, 'b3': 0.51, 'AUC': 0.641}
   a=1  {'tok85': 902.61, 'tok90': 964.67, 'tok95': 1044.93, 'b3': 0.51, 'AUC': 0.646} | tok90 Δ (-5.5, 13.9) | tok85 Δ (-14.2, 16.9)
   a=3  {'tok85': 969.67, 'tok90': 1032.9, 'tok95': 1103.86, 'b3': 0.52, 'AUC': 0.636} | tok90 Δ (57.2, 35.3) | tok85 Δ (34.3, 53.6)
a=3 within-q sd(Δs)=1.598 | B hard-minus-beating Δs 0.992 (frac>0 0.78) | ratio to a=1 Δs (B, regression-free: sd ratio) 10.36

```
Passes left: 0/5. Turn 27/40. The forward-pass budget is used up. Code still runs (forward() is refused). Make sure your current removals are your final edit, then reply WITHOUT a code block with your final explanation (mechanism, changes, evidence).