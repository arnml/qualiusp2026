# Notes

Session start. Baseline cost: {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}

## Split: rng(0) permutation; A=perm[:100] selection, B=perm[100:] confirmation.
## Pass1: original diagnostics. bg3 A .91 B .98 all .945. Decision built in L4-L5 (CLS mean-diff dir bg3: L3out .25, L4out .50, final .98).
## Pass2: drop L1 + L5 1300 neurons (ranked by within-q var) + L5 heads 11,5,8 -> FAIL (A -7pp, B -17pp, spearman .76).
  - L1 drop damages token reps: L4-out CLS discriminability .70->.51 though CLS cos at L2 out .997 (CLS insensitive early).
  - L5 var-ranked neurons were GELU-floor constants: removed ~44% of LN-scale as constant offset -> damage. Use RMS ranking instead.
## Pass3 plan: L5 RMS-700 + L5 heads 11,5 + 140 CLS-silent early neurons (L0-3, max|a|<.005).

## Pass3 result: A lost1/gained1 (net 0), B 0 flips; spearman .987/.988; within-q score-diff std .25. cost 36.660 GF.
## Pass4 (P3 + L5 RMS-1000, L5h8, early max|a|<.02 neurons, heads L4h9,L1h6,L1h7): A net -1 (q13 lost), B 0; std .55 -> too noisy. cost 34.69.
## Noise model (Gaussian noise on original scores): std .25 -> -0.31pp exp; .55 -> -0.55pp, P(<-1pp)=.20 on 200 q.
## Pass5 (P3 + L5 RMS-1000, L5h8, L4h9): A net -1 (q13 again), B 0; std .398. cost 35.47.
## FINAL: revert to P3 (pre-registered rule: adopt P5 only if no net loss). 36.660 GF (-6.6%).
