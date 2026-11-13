The coverage edit points the right way but is about 25× too weak:
- It favours hard golds over the distractors that beat them on A (Δ +0.10, 76% of questions) and replicates on held-out B (Δ +0.14, 67%).
- It anti-correlates with overlap (−0.34), as designed.
- But its within-question spread is only 0.15 against a score spread of 3.9.

Extrapolating the measured change, AUC rises monotonically on both splits (A 0.720 → 0.731 at 3×, B 0.641 → 0.657). B token counts are noisy and not clearly better, so a real gain on the main metric is uncertain.

The L5 body-head edit, by contrast, moved scores toward title-matched paragraphs (corr 0.60) but against hard golds, consistent with its null result.

Last pass: a dose-response point for the same 87 neurons at α = 3, i.e. their output columns × (−2). Strength 3 is the moderate end of the A extrapolation. I'll choose between α = 1 and α = 3 using A, report B, and revert if neither clearly helps.

```python
# Hypothesis: (pass 5, dose-response) the same 87 L4 coverage neurons at alpha=3 (columns x -2) give ~3x the alpha=1 shift toward bridge golds.
S_cov = np.load(WORK / "S_cov_L4.npy")
def cov_edit(alpha):
    W = load_original("bert.encoder.layer.4.output.dense.weight").clone(); W[:, torch.as_tensor(S_cov)] *= (1 - alpha); return W
torch.save(cov_edit(1.0), WORK / "edit_cov_a1.pt"); torch.save(cov_edit(3.0), WORK / "edit_cov_a3.pt")
save_edit("bert.encoder.layer.4.output.dense.weight", cov_edit(3.0)); print(edited().get("raw_changed_entries"))
out5 = forward(items=items, model="edited")
S5 = np.asarray(out5["scores_edited"], float); np.save(WORK / "scores_cov_a3.npy", S5); print("saved; passes_left:", out5["passes_left"])
for sp, qs in [("A", qA), ("B", qB)]:
    print(sp, "orig", ev(S0, qs)); print("   a=1 ", ev(S4, qs), "| tok90 Δ", boot(S4, qs), "| tok85 Δ", boot(S4, qs, rc=0.85))
    print("   a=3 ", ev(S5, qs), "| tok90 Δ", boot(S5, qs), "| tok85 Δ", boot(S5, qs, rc=0.85))
d5 = S5 - S0
gB = np.array([d5[(qi_arr == q) & hardmask].mean() - d5[(qi_arr == q) & beatmask].mean() for q in qB if ((qi_arr == q) & beatmask).any()])
print("a=3 within-q sd(Δs)=%.3f | B hard-minus-beating Δs %.3f (frac>0 %.2f) | ratio to a=1 Δs (B, regression-free: sd ratio) %.2f" % (cq(d5).std(), gB.mean(), (gB > 0).mean(), cq(d5).std() / cq(S4 - S0).std()))
```