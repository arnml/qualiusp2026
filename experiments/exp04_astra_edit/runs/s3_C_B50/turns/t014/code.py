# Hypothesis: isolated low-strength heads can be dispensable even when removing three heads from the same layer is harmful.
head_test_parent = {
    "neurons": configs["MLP_balanced2"]["neurons"],
    "heads": [(1, h) for h in head_ranks[1][:3]],
    "layers": [],
}
singleton_results = {}
for l in [0, 2, 3, 4, 5]:
    h = head_ranks[l][0]
    label = f"balanced2_heads1_plus_{l}_{h}"
    configs[label] = {"neurons": head_test_parent["neurons"],
                      "heads": head_test_parent["heads"] + [(l, h)], "layers": []}
    apply_config(**configs[label])
    out = forward(items=select_items, model="both")
    outputs[label] = np.asarray(out["scores_edited"]).copy()
    singleton_results[(l, h)] = report(label, out)
reset_all()
