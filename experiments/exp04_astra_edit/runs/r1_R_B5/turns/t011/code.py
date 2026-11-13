# Hypothesis: complete removal of the selected lexical outputs may outperform both partial attenuation and sign reversal.
install_attenuation(1.0)
trial_zero = forward(items=discovery_items, model="both", cls=True)
zero_metrics, zero_report = paired_report(trial_zero["scores_edited"])
trials[1.0] = {"output": trial_zero, "metrics": zero_metrics, "report": zero_report}
print("Strength 1.0:", zero_report)
print("Paired uncertainty:",
      paired_bootstrap(trial_zero["scores_original"], trial_zero["scores_edited"],
                       groups_d, dgold, dlength))
print("Discovery comparison:",
      {a: {"both@3": t["metrics"]["both@3"], "tokens@90": t["metrics"]["tokens@90"]}
       for a, t in sorted(trials.items())})
with NOTES.open("a") as f:
    f.write("Strength 1.0: " + repr(zero_report) + "\n")
    f.write("Caution: stronger attenuation did not monotonically improve top3 or token retention. "
            "Observed score movement alone is not evidence of ranking repair.\n")
