OUTPUT:
```
cos(len, r)=-0.93 cos(title, r)=0.93 cos(len_o, ti_o)=-0.74
len_o    B: corr(log ntok)=0.33 corr(title)=-0.33 corr(score)=-0.89 corr(ov)=-0.47 | B gold-sep=-1.01 sd | within-q sd=0.718 (score-axis sd 2.301)
title_o  B: corr(log ntok)=-0.30 corr(title)=0.39 corr(score)=0.84 corr(ov)=0.39 | B gold-sep=0.98 sd | within-q sd=0.563 (score-axis sd 2.301)
     dir  beta  tok85  tok90  tok95   b3  AUC_hard_vs_distr_pooled
  -len_o  0.25  852.1  952.0 1032.1 0.58                     0.719
  -len_o  0.50  862.7  955.9 1035.8 0.59                     0.717
  -len_o  1.00  816.7  964.3 1041.1 0.60                     0.716
+title_o  0.25  840.8  955.4 1041.7 0.57                     0.720
+title_o  0.50  844.7  975.4 1048.5 0.58                     0.721
+title_o  1.00  845.7  982.0 1060.4 0.60                     0.721

```
Passes left: 4/5. Turn 16/40.