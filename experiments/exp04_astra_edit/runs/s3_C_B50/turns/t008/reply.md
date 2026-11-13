Removing either whole layer hurt quality, so both are restored. Removing the weakest quarter of MLP neurons by weight strength preserved every original success on the 120 selection questions and added two, saving 15.3% of encoder FLOPs. I’ll test compensation and deeper cuts before checking the reserved questions.


```python
# Hypothesis: modest output-column rescaling can compensate for the aggregate signal removed by MLP pruning.
scaling_results = {}
for factor in [1.10, 1.25]:
    label = f"MLP_weight_25_scale_{factor}"
    cfg = configs["MLP_weight_25"]
    apply_config(**cfg)
    for l in range(6):
        kept = sorted(set(range(1536)) - set(cfg["neurons"][l]))
        scale_cols(wn(l, "output.dense.weight"), kept, factor)
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    scaling_results[factor] = report(label, out)
reset_all()
with NOTES.open("a") as f:
    f.write("\nWeight salience uses incoming row-plus-bias L2 norm times outgoing column L2 norm. "
            "At 25% per layer it preserves all 111 original selection successes and adds two. "
            "Testing two fixed, uniform compensation constants (1.10 and 1.25); no fitting or optimization.\n")
```
