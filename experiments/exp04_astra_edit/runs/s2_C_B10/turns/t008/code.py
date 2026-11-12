# Hypothesis: eight weak or structurally redundant heads can be removed on top of the validated MLP candidate.
head_candidates = [(0, 2), (1, 7), (1, 8), (2, 11), (3, 1), (4, 9), (5, 9), (5, 11)]
configs["mlp_A_heads8"] = copy.deepcopy(configs["mlp_A"])
configs["mlp_A_heads8"]["heads"] = head_candidates

with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nHead candidate (zero-based): L0H2 has weak weights and CLS attention almost identical "
            "to a retained structural head; L1H7/H8 have weak static readout norms and highly redundant "
            "CLS attention; L2H11 and L3H1 have the weakest value/output maps in their layers; "
            "L4H9 has the weakest map and static readout norm in layer 4; "
            "L5H9 has the weakest map and overlaps H4's question-focused attention; "
            "L5H11 has the weakest static readout norm and much structural attention. "
            "These are hypotheses to test causally, not claims of proven redundancy.\n")
apply_config(configs["mlp_A_heads8"])
head_trial = forward(items=discovery_items, model="both")
report_trial("mlp_A_heads8", head_trial)
