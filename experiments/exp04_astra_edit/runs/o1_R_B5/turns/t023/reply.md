Title neurons are a dead end. Their attributed leverage on the relevance axis is about 0.00–0.02, against a layer-4 MLP total of −0.61 (A) / −0.70 (B). Title information is present at layer 4 but is not written onto the axis the score reads.

The leverage is in the layer-4 MLP's **coverage** neurons. These are general lexical-relevance units at [CLS] (corr with body overlap 0.45). They fire more for the high-overlap distractors than for the bridge golds those distractors beat, and the top 40 alone account for about half of the layer's hard-vs-beating deficit (−0.32 of −0.61 on A, −0.35 on B). The split-A diagnostic agrees: down-weighting overlap was one of the few re-weightings that helped tok85/tok90.

Pass 3 plan:
- Drop the null L5 head edit.
- Select layer-4 neurons by their split-A hard − beating attribution: take the smallest set covering 60% of the negative mass, capped at 250.
- Zero their output columns (α = 1).
- Evaluate on A and B with bootstrap SEs, per question type.

```python
# Hypothesis: L4 [CLS] lexical-coverage neurons over-reward high-overlap distractors; removing them (alpha=1) lifts bridge golds over beating distractors.
drop_edit("bert.encoder.layer.5.attention.output.dense.weight")
W4 = load_original("bert.encoder.layer.4.output.dense.weight").numpy(); g4 = load_original("bert.encoder.layer.4.output.LayerNorm.weight").numpy()
w4 = tab[4].w.values
cA4 = paired_gap(M[:, 4], hardmask, beatmask, qA).mean(0) * w4; cB4 = paired_gap(M[:, 4], hardmask, beatmask, qB).mean(0) * w4
hdA4 = paired_gap(M[:, 4], hardmask, dist, qA).mean(0) * w4; hdB4 = paired_gap(M[:, 4], hardmask, dist, qB).mean(0) * w4
order = np.argsort(cA4); negmass = -cA4[cA4 < 0].sum(); cum = np.cumsum(-cA4[order])
k = int(min(250, np.searchsorted(cum, 0.6 * negmass) + 1)); S_cov = order[:k]
print("k=%d neurons | hard-beating attrib removed: A %.3f (of total %.3f) B %.3f (of %.3f) | hard-all distr attrib: A %.3f B %.3f"
      % (k, cA4[S_cov].sum(), cA4.sum(), cB4[S_cov].sum(), cB4.sum(), hdA4[S_cov].sum(), hdB4[S_cov].sum()))
np.save(WORK / "S_cov_L4.npy", S_cov)
def cov_edit(alpha):
    W = load_original("bert.encoder.layer.4.output.dense.weight").clone(); W[:, torch.as_tensor(S_cov)] *= (1 - alpha); return W
save_edit("bert.encoder.layer.4.output.dense.weight", cov_edit(1.0)); print(edited().get("raw_changed_entries"))
out3 = forward(items=items, model="edited", cls=True)
S3 = np.asarray(out3["scores_edited"], float); np.save(WORK / "scores_cov_a1.npy", S3); np.save(WORK / "cls_cov_a1.npy", np.asarray(out3["cls_edited"])); print("passes_left:", out3["passes_left"])
for sp, qs in [("A", qA), ("B", qB)]:
    print(sp, "orig", ev(S0, qs), "\n   cov a=1", ev(S3, qs), "| tok90 Δ", boot(S3, qs), "| tok85 Δ", boot(S3, qs, rc=0.85))
for t in ["bridge", "comparison"]:
    qs = [q for q in qB if practice[q]["type"] == t]; print("B", t, len(qs), "orig", ev(S0, qs)["AUC"], "->", ev(S3, qs)["AUC"])
```