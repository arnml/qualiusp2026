# Hypothesis: layer 1 may be removable because its CLS update is small and has little direct relevance contrast.
import copy
import json

experiments = {}
configs = {}

def apply_config(config):
    reset_all()
    for layer in config.get("layers", []):
        remove_layer(int(layer))
    for layer, head in config.get("heads", []):
        remove_head(int(layer), int(head))
    for layer, indices in config.get("neurons", {}).items():
        if len(indices):
            remove_neurons(int(layer), [int(i) for i in indices])
    for name, cols, factor in config.get("scales", []):
        scale_cols(name, [int(i) for i in cols], float(factor))

def report_trial(name, result, items=discovery_items):
    original_scores = result.get("scores_original", baseline["scores_original"])
    original = evaluate(original_scores, items)
    candidate = evaluate(result["scores_edited"], items)
    losses = [q for q in original if original[q]["ok"] and not candidate[q]["ok"]]
    gains = [q for q in original if not original[q]["ok"] and candidate[q]["ok"]]
    deltas = [candidate[q]["margin"] - original[q]["margin"] for q in original
              if np.isfinite(original[q]["margin"])]
    summary = {
        "name": name, "original": sum(v["ok"] for v in original.values()),
        "edited": sum(v["ok"] for v in candidate.values()), "n": len(original),
        "losses": losses, "gains": gains,
        "margin_delta_quantiles": np.round(np.quantile(deltas, [.05, .25, .5, .75, .95]), 3).tolist(),
        "score_correlation": round(float(np.corrcoef(original_scores, result["scores_edited"])[0, 1]), 5),
        "cost": cost(),
    }
    experiments[name] = {"summary": summary, "evaluation": candidate,
                         "scores": np.asarray(result["scores_edited"]).copy()}
    print(summary)
    with NOTES.open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps(summary) + "\n")
    return summary

configs["drop_layer1"] = {"layers": [1], "heads": [], "neurons": {}}
apply_config(configs["drop_layer1"])
layer1_trial = forward(items=discovery_items, model="both")
report_trial("drop_layer1", layer1_trial)
