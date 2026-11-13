OUTPUT:
```
within-q corr(score, log ntok): distractors -0.27 | golds -0.08 | all -0.30
within-q corr(overlap, log ntok) distractors 0.03; corr(score, log ntok | overlap-partialled, distr) -0.32
gold vs distractor log-ntok gap (within-q, A/B): -0.27 / -0.27
L1 length-dir: corr(log ntok) B=0.96 | corr(score) B=-0.29 | cos(m6)=-
L2 length-dir: corr(log ntok) B=0.93 | corr(score) B=-0.27 | cos(m6)=-
L3 length-dir: corr(log ntok) B=0.96 | corr(score) B=-0.33 | cos(m6)=-
L4 length-dir: corr(log ntok) B=0.94 | corr(score) B=-0.40 | cos(m6)=-
L5 length-dir: corr(log ntok) B=0.49 | corr(score) B=-0.91 | cos(m6)=-
L6 length-dir: corr(log ntok) B=0.28 | corr(score) B=-0.98 | cos(m6)=-0.98
L5 h0: CLS->self 0.02  ->[SEP]s 0.03  ->question 0.08  ->title 0.05  ->body 0.82
L5 h6: CLS->self 0.01  ->[SEP]s 0.01  ->question 0.05  ->title 0.13  ->body 0.80
L5 h11: CLS->self 0.15  ->[SEP]s 0.37  ->question 0.07  ->title 0.04  ->body 0.37
L4 h1: CLS->self 0.22  ->[SEP]s 0.39  ->question 0.14  ->title 0.01  ->body 0.25
L4 h4: CLS->self 0.25  ->[SEP]s 0.48  ->question 0.12  ->title 0.02  ->body 0.13
L4 h10: CLS->self 0.07  ->[SEP]s 0.13  ->question 0.18  ->title 0.03  ->body 0.59
L4 h11: CLS->self 0.06  ->[SEP]s 0.12  ->question 0.10  ->title 0.03  ->body 0.68

```
Passes left: 4/5. Turn 15/40.