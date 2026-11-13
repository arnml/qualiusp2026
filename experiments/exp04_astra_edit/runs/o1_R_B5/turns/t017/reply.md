The layer-6 [CLS] has no usable extra information. The "orthogonal-to-readout" length and title directions still correlate 0.84–0.89 with the score, and adding them gives no proxy gain (AUC stuck around 0.72). So pooler/readout edits are a dead end; the fix has to change what layer 5 computes.

Pass 2 is a causal test of the head hypothesis: amplify the two body-reading heads in layer 5 (h0, h6) ×2 by scaling their 32 columns of `attention.output.dense`. That is 24,576 weights. I'm also recording the edited [CLS] states, so the measured change shows what these heads write at [CLS].

```python
# Hypothesis: final-layer coverage heads dominate [CLS]; amplifying the paragraph-body heads L5.h0/h6 (x2) lifts answer-bearing bridge golds vs distractors.
save_edit("bert.encoder.layer.5.attention.output.dense.weight", head_scale_edit(5, {0: 2.0, 6: 2.0})); print(edited())
out2 = forward(items=items, model="edited", cls=True)
S2 = np.asarray(out2["scores_edited"], float); C2 = np.asarray(out2["cls_edited"]); print("passes_left:", out2["passes_left"])
np.save(WORK / "scores_L5h0h6x2.npy", S2); np.save(WORK / "cls_L5h0h6x2.npy", C2)
def compare(Snew, label):
    for sp, mask, Fs in [("A", A, FA), ("B", ~A, FB)]:
        o, e = robust(Fs, S0[mask]), robust(Fs, Snew[mask])
        b0, b1 = metrics(Fs, S0[mask])[1], metrics(Fs, Snew[mask])[1]; d = b1.astype(int) - b0.astype(int)
        print("%s %s orig %s\n      edit %s | b3 per-q +%d/-%d" % (label, sp, o, e, (d > 0).sum(), (d < 0).sum()))
compare(S2, "h0h6x2")
print("within-q corr(Δscore, has_ans)=%.2f title=%.2f body_ov=%.2f log-ntok=%.2f" % tuple(np.corrcoef(cq(S2 - S0), cq(F[f].values.astype(float)))[0,1] for f in ["has_ans", "title_in_q", "body_ov"]) + (np.corrcoef(cq(S2 - S0), lnt)[0,1],))
```