Pass 5 lands on the boundary:
- **Fidelity:** within-question score std 0.398 (right at the budget) and Spearman 0.981 to 0.985.
- **Split B:** unchanged.
- **Split A:** still nets −1. Question 13 is lost, the same flip as in pass 4, so it comes from a component the two passes share (extra layer-5 neurons, L5h8 or L4h9).

My rule set before the pass required no net loss, and pass 5 fails it on the selection half. I'll revert to pass 3, the tested clean configuration, and verify the state matches it exactly.

```python
# Hypothesis: none (finalization). Restore exactly the P3 edit (tested clean: A net 0, B 0 flips, within-q std 0.25) and verify cost/state.
reset_all()
remove_neurons(5, L5_rms)                 # 700 lowest-RMS-contribution L5 neurons (ranked on split A)
for h in [11,5]: remove_head(5,h)
for l,js in silent.items(): remove_neurons(l,js)   # 140 neurons with max|output|<0.005 at CLS on all items
c=cost(); print(c); print(edited())
assert abs(c['encoder_gflops_per_question']-36.6603776)<1e-6
NOTES.write_text(NOTES.read_text()+"""
## Pass3 result: A lost1/gained1 (net 0), B 0 flips; spearman .987/.988; within-q score-diff std .25. cost 36.660 GF.
## Pass4 (P3 + L5 RMS-1000, L5h8, early max|a|<.02 neurons, heads L4h9,L1h6,L1h7): A net -1 (q13 lost), B 0; std .55 -> too noisy. cost 34.69.
## Noise model (Gaussian noise on original scores): std .25 -> -0.31pp exp; .55 -> -0.55pp, P(<-1pp)=.20 on 200 q.
## Pass5 (P3 + L5 RMS-1000, L5h8, L4h9): A net -1 (q13 again), B 0; std .398. cost 35.47.
## FINAL: revert to P3 (pre-registered rule: adopt P5 only if no net loss). 36.660 GF (-6.6%).
""")
```