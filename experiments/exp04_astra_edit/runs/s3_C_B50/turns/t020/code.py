# Hypothesis: the final MLP mainly calibrates scores, whereas final-layer attention may still be necessary for relevance ranking.
for l in range(7):
    projection = np.asarray(diag["cls_original"][:, l], dtype=np.float32) @ readout
    print("CLS stage", l, "correlation with original score:",
          round(float(np.corrcoef(projection, base_s)[0, 1]), 4))

last_layer_control = {
    "neurons": {l: list(final_cfg["neurons"][l]) for l in range(5)},
    "heads": [], "layers": [5],
}
apply_config(**last_layer_control)
out = forward(items=select_items, model="both")
last_layer_row = report("MECHANISTIC_CONTROL_remove_layer5", out)
apply_config(**final_cfg)
with NOTES.open("a") as f:
    f.write("\nUnexpected control result: removing all final-MLP neurons also preserves all original "
            "selection successes (112/120). This falsifies the hypothesis that its retained 154 "
            "neurons are necessary for selection ranking. Testing whether final attention is needed, "
            "then verifying a whole-MLP simplification if appropriate.\n")
