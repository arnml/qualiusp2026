# Hypothesis: (finalize) no tested edit beats its held-out SE -> revert to original weights; record everything in NOTES.
reset_all(); print("edited():", edited())
with open(NOTES, "a") as fh:
    fh.write("""
## Results (A chooses, B confirms; tok90 delta = paired bootstrap over questions, 500 resamples)
- P4 cov alpha=1 (87 cols x0): A tok90 -14.9+-19.3, AUC .720->.724 | B tok90 -5.5+-13.9, tok85 -14.2+-16.9, AUC .641->.646, b3 .51->.51.
  Targeted contrast (hard gold minus beating distractor, score change): A +.10 (76% q>0), B +.14 (67%).
- P5 cov alpha=3 (87 cols x-2): A tok90 +10+-63, AUC .716 | B tok90 +57.2+-35.3, AUC .636, b3 .52.
  Targeted contrast B +.99 (78% q>0): strong dose-response on the mechanism, but net ranking worse (nonlinear, 10x score change).
- P2 L5 h0/h6 x2: null (B tok90 -8.8+-14.5); score change tracks title match (.60) but slightly hurts hard-vs-beating.
## Decision
No edit's held-out gain exceeds its SE; alpha=3 is harmful. Final submission: reset_all() (original weights).
Candidate tensors kept: edit_cov_a1.pt, edit_cov_a3.pt.
""")
print(NOTES.read_text()[-600:])
