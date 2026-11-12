# Hypothesis: additional weak MLP paths are safer to remove than the attention heads rejected by the causal test.
plans["C_more_mlp"] = make_plan([640, 768, 768, 512, 640, 1152])
apply_plan(plans["C_more_mlp"])
print("Candidate C cost:", cost())

# Selection rows come first; confirmation results remain unexamined.
evaluation_items = baseline_items + confirmation_items
results["C_more_mlp_full"] = forward(items=evaluation_items, model="both", cls=True)
n_selection_rows = len(baseline_items)
results["C_more_mlp_selection"] = {
    key: results["C_more_mlp_full"][key][:n_selection_rows]
    for key in ("scores_original", "scores_edited", "cls_original", "cls_edited")
}
report_c = compare_result(baseline_items, results["C_more_mlp_selection"])
print("Candidate C selection:", report_c)

c_orig = results["C_more_mlp_selection"]["scores_original"].copy()
c_edit = results["C_more_mlp_selection"]["scores_edited"].copy()
for q in selection_q:
    ix = np.array([i for i, item in enumerate(baseline_items) if item[0] == q])
    c_orig[ix] -= c_orig[ix].mean()
    c_edit[ix] -= c_edit[ix].mean()
c_correlation = float(np.corrcoef(c_orig, c_edit)[0, 1])
print("Candidate C within-question score correlation:", round(c_correlation, 4))

# Fix the preferred candidate using selection data alone.
# Require at least the original success count, at most two regressions,
# and strong continuous-score agreement.
prefer_c = (
    report_c["edited_successes"] >= report_c["original_successes"]
    and len(report_c["losses"]) <= 2
    and c_correlation >= 0.965
)
preferred_candidate = "C_more_mlp" if prefer_c else "A_mlp"
print("Selection-only preferred candidate:", preferred_candidate)
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nCandidate B rejected: 88/100 versus 92/100 original; all heads restored.\n")
    f.write("Candidate C MLP removals [640,768,768,512,640,1152], no heads or layers removed. "
            "Incremental pruning emphasizes earlier layers with very small observed CLS drift under A; "
            "this remains a hypothesis because other-token activations are unavailable.\n")
    f.write("Pass 4 selection result: " + str(report_c) + "\n")
    f.write("Within-question score correlation: " + str(c_correlation) + "\n")
    f.write("Selection-only preferred candidate, fixed before confirmation inspection: " + preferred_candidate + "\n")
    f.write("Confirmation arrays collected but no confirmation statistics inspected.\n")
