The weight check points to layer 4: it contains a concentrated group of neurons whose outputs favor score increases and whose activations favor lexical distractors over weak gold paragraphs. I’ll test attenuation of the 96 strongest candidates, starting at 50%, and measure paired changes across the discovery questions.

```python
# Hypothesis: attenuating the lexical-distractor-promoting outputs in layer 4 will improve gold retention and ranking.
target_name = "bert.encoder.layer.4.output.dense.weight"
target_original = load_original(target_name).clone()
layer4_diagnostic = next(x for x in diagnostic_layers if x["layer"] == 4)
target_neurons = np.argsort(-layer4_diagnostic["priority"])[:96].copy()
assert (layer4_diagnostic["priority"][target_neurons] > 0).all()
target_columns = torch.tensor(target_neurons.tolist(), dtype=torch.long)

def install_attenuation(strength):
    candidate = target_original.clone()
    candidate[:, target_columns] = target_original[:, target_columns] * (1.0 - strength)
    save_edit(target_name, candidate)

def paired_report(scores):
    result = metrics(np.asarray(scores), groups_d, dgold, dlength)
    delta = result["success"].astype(float) - base_d["success"]
    return result, {
        "both@3": result["both@3"],
        "tokens@90": result["tokens@90"],
        "top3_gain_pp": float(100 * delta.mean()),
        "paired_top3_se_pp": float(100 * delta.std(ddof=1) / np.sqrt(len(delta))),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "weak_gold_vs_second_distractor_margin_change": float(
            ((scores[weak_gold] - scores[competing_distractor]) -
             (dscores[weak_gold] - dscores[competing_distractor])).mean())
    }

install_attenuation(0.5)
with NOTES.open("a") as f:
    f.write("\nMechanism candidate: layer 4 MLP outputs promoting lexical distractors. "
            "Screened using within-question activation correlations, distractor-minus-weak-gold means, "
            "and output-weight alignment with the layer's score-associated CLS direction. "
            "These are descriptive moments, not a fitted objective.\n")
    f.write("Fixed selected columns: " + repr(target_neurons.tolist()) + "\n")
    f.write("Intervention W[:, S] = (1-alpha) W_original[:, S]; 36,864 affected entries.\n")
trial_half = forward(items=discovery_items, model="both", cls=True, mlp=True)
half_metrics, half_report = paired_report(trial_half["scores_edited"])
trials = {0.5: {"output": trial_half, "metrics": half_metrics, "report": half_report}}
print("Strength 0.5:", half_report)
print("Paired baseline agreement:", float(np.max(np.abs(trial_half["scores_original"] - dscores))))
print("Selected columns:", target_neurons.tolist())
print("Edit status:", edited())
with NOTES.open("a") as f:
    f.write("Strength 0.5: " + repr(half_report) + "\n")
```
