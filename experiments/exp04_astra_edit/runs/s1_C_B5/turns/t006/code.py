# Hypothesis: removing weak MLP paths preserves ranking while reducing roughly one quarter of encoder work.
neuron_order = {}
for layer in range(6):
    rms = np.sqrt(np.mean(baseline["mlp_original"][:, layer, :].astype(np.float64)**2, axis=0))
    output_norm = weight_stats[layer]["mlp_output_norm"]
    if layer < 5:
        input_norm = weight_stats[layer]["mlp_input_norm"]
        # Weight strength protects paths that may be useful at paragraph tokens despite weak CLS activity.
        importance = output_norm * input_norm * np.sqrt(0.25 + rms / max(float(np.median(rms)), 1e-8))
    else:
        # After the final MLP, only CLS is consumed by the scoring head.
        importance = output_norm * rms
    neuron_order[layer] = np.argsort(importance)

plans = {}
results = {}
def make_plan(counts, heads=(), layers=()):
    return {
        "neurons": {l: neuron_order[l][:int(n)].astype(int).tolist() for l, n in enumerate(counts) if n},
        "heads": list(heads), "layers": list(layers)
    }

def apply_plan(plan):
    reset_all()
    for l in plan["layers"]:
        remove_layer(l)
    for l, h in plan["heads"]:
        remove_head(int(l), int(h))
    for l, indices in plan["neurons"].items():
        if l not in plan["layers"]:
            remove_neurons(int(l), indices)

def compare_result(items, result, subset=None):
    original = question_metrics(items, result["scores_original"])
    changed = question_metrics(items, result["scores_edited"])
    qs = sorted(original) if subset is None else sorted(set(original) & set(subset))
    losses = [q for q in qs if original[q]["success"] and not changed[q]["success"]]
    gains = [q for q in qs if not original[q]["success"] and changed[q]["success"]]
    finite = [q for q in qs if np.isfinite(original[q]["margin"]) and np.isfinite(changed[q]["margin"])]
    margin_change = np.array([changed[q]["margin"]-original[q]["margin"] for q in finite])
    report = {
        "n": len(qs),
        "original_successes": sum(original[q]["success"] for q in qs),
        "edited_successes": sum(changed[q]["success"] for q in qs),
        "losses": losses, "gains": gains,
        "margin_change_q10_median_q90": np.round(np.quantile(margin_change, [.1, .5, .9]), 4).tolist(),
        "by_type": {
            typ: {
                "n": sum(practice[q]["type"] == typ for q in qs),
                "original": sum(original[q]["success"] for q in qs if practice[q]["type"] == typ),
                "edited": sum(changed[q]["success"] for q in qs if practice[q]["type"] == typ),
            } for typ in sorted({practice[q]["type"] for q in qs})
        }
    }
    return report

plans["A_mlp"] = make_plan([384, 512, 512, 512, 640, 1024])
apply_plan(plans["A_mlp"])
print("Candidate A cost:", cost())
results["A_mlp"] = forward(items=baseline_items, model="both", cls=True, mlp=True)
report_a = compare_result(baseline_items, results["A_mlp"])
print("Candidate A selection:", report_a)
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nCandidate A: MLP removals per layer [384,512,512,512,640,1024]; no heads/layers removed.\n")
    f.write("Ranking for layers 0-4: output-column norm * input-row norm * sqrt(0.25 + CLS RMS/median RMS). "
            "Final-layer ranking: output-column norm * CLS RMS. All statistics use selection only.\n")
    f.write("Pass 2 paired selection result: " + str(report_a) + "\nCost: " + str(cost()) + "\n")
