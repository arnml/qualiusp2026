```python
# Hypothesis: retaining only a statistically supported confirmation result prevents an apparent discovery gain from becoming an unsupported weight edit.
if keep_candidate:
    final_action = "Kept the confirmed alpha=1.0 edit."
else:
    drop_edit(target_name)
    final_action = "Restored the original weights because the confirmation criterion was not met."

final_status = edited()
final_changed_entries = int(torch.count_nonzero(load(target_name) != load_original(target_name)).item())
np.savez(
    WORK / "experiment_results.npz",
    discovery_ids=discovery_ids,
    confirmation_ids=confirmation_ids,
    target_neurons=target_neurons,
    original_discovery_scores=dscores,
    half_discovery_scores=trials[0.5]["output"]["scores_edited"],
    zero_discovery_scores=trials[1.0]["output"]["scores_edited"],
    double_discovery_scores=trials[2.0]["output"]["scores_edited"],
    original_confirmation_scores=confirmation["scores_original"],
    candidate_confirmation_scores=confirmation["scores_edited"]
)
with NOTES.open("a") as f:
    f.write("\nFinal action: " + final_action + "\n")
    f.write("Final changed entries in tested tensor: " + str(final_changed_entries) + "\n")
    f.write("Final edited(): " + repr(final_status) + "\n")
    f.write("Interpretation: discovery showed an overlap-associated signal, but causal ranking gains "
            "must be distinguished from score shifts and sampling-sensitive threshold changes. "
            "The final submission follows the untouched confirmation result.\n")
print(final_action)
print("Final changed entries:", final_changed_entries)
print("Final edited():", final_status)
print("Notes and reproducibility artifacts are saved in WORK.")
```
