# Hypothesis: a useful edit must reduce token retention beyond uncertainty from both question sampling and threshold calibration.
def paired_bootstrap(original_scores, edited_scores, groups, gold, lengths, repetitions=2000, seed=7301):
    original_scores = np.asarray(original_scores)
    edited_scores = np.asarray(edited_scores)
    n = len(groups)
    draws = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    outputs = []
    for scores in [original_scores, edited_scores]:
        result = metrics(scores, groups, gold, lengths)
        thresholds = np.sort(result["min_gold"][draws], axis=1)[:, int(np.floor(0.1*n))]
        all_costs = np.empty((repetitions, n))
        for j, ix in enumerate(groups):
            keep = scores[ix][None, :] >= thresholds[:, None]
            empty = ~keep.any(axis=1)
            keep[empty, np.argmax(scores[ix])] = True
            all_costs[:, j] = keep @ lengths[ix]
        sampled_cost = all_costs[np.arange(repetitions)[:, None], draws].mean(axis=1)
        sampled_success = result["success"][draws].mean(axis=1)
        outputs.append((sampled_cost, sampled_success))
    token_saving = outputs[0][0] - outputs[1][0]
    top3_gain = 100 * (outputs[1][1] - outputs[0][1])
    return {
        "token_saving_se": float(token_saving.std(ddof=1)),
        "token_saving_95pct_interval": np.quantile(token_saving, [0.025, 0.975]).tolist(),
        "top3_gain_se_pp": float(top3_gain.std(ddof=1)),
        "top3_gain_95pct_interval_pp": np.quantile(top3_gain, [0.025, 0.975]).tolist()
    }

print("Bootstrap helper ready; it recalibrates both thresholds in every paired resample.")
if "half_report" in globals():
    print("Half-strength report:", half_report)
    print("Half-strength uncertainty:",
          paired_bootstrap(dscores, trial_half["scores_edited"], groups_d, dgold, dlength))
