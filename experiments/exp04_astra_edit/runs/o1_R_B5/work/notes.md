# Notes: cross-encoder weight edit for multi-hop gold ranking
Split: A = even question index (choose), B = odd (confirm). Metrics on practice: tok@85/90/95 (tok length proxy), both-gold@3, pooled AUC hard-gold vs distractors.
Baseline: A tok90 945.5 b3 .58 AUC .720 | B tok90 963.1 b3 .51 AUC .641.
## Diagnosis (pass 1, original, all 1994 paragraphs, cls+mlp+attn)
- Hard (min-score) gold in bridge questions = answer-bearing 2nd-hop paragraph: has_ans .80, overlap .42, title_in_q .34, rank 4.5.
- Score within question tracks lexical match: corr overlap .60, title .46. Final [CLS] ~1-D (gold dir cos .96 with readout); no unused gold direction at L6 -> pooler edits pointless.
- Last layer: 7/12 heads at [CLS] read question tokens (coverage); h0/h6 read paragraph body.
- Layer trace along per-layer relevance axis (B): hard-minus-beating gap +.40sd (C2) -> +.08 (C3) -> -.22 (C4) -> -.61 (C5) -> -.66 (C6); layers 2-4 build coverage that favours high-overlap distractors.
- L4 MLP direct attribution to hard-minus-beating: -0.61 (A) / -0.70 (B); per-neuron attributions replicate (corr .98). Anti-hard neurons = coverage (corr body_ov .45).
- Length: model already penalises length (corr -.27); not used (dataset artifact).
- Title neurons: replicate but ~0 leverage on relevance axis.
## Passes
- P2: L5 heads 0,6 x2 -> null (B tok90 -8.8 +- 14.5). Dropped.
- P3: lost to kernel timeout.
- P4: L4 coverage neurons (87, chosen on A: 60% of negative hard-beating attribution) output columns x0.

## Results (A chooses, B confirms; tok90 delta = paired bootstrap over questions, 500 resamples)
- P4 cov alpha=1 (87 cols x0): A tok90 -14.9+-19.3, AUC .720->.724 | B tok90 -5.5+-13.9, tok85 -14.2+-16.9, AUC .641->.646, b3 .51->.51.
  Targeted contrast (hard gold minus beating distractor, score change): A +.10 (76% q>0), B +.14 (67%).
- P5 cov alpha=3 (87 cols x-2): A tok90 +10+-63, AUC .716 | B tok90 +57.2+-35.3, AUC .636, b3 .52.
  Targeted contrast B +.99 (78% q>0): strong dose-response on the mechanism, but net ranking worse (nonlinear, 10x score change).
- P2 L5 h0/h6 x2: null (B tok90 -8.8+-14.5); score change tracks title match (.60) but slightly hurts hard-vs-beating.
## Decision
No edit's held-out gain exceeds its SE; alpha=3 is harmful. Final submission: reset_all() (original weights).
Candidate tensors kept: edit_cov_a1.pt, edit_cov_a3.pt.
