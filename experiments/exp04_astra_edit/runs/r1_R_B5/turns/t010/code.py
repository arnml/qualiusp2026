# Hypothesis: a causal repair should improve low-tail gold ranks and reduce lexical preference consistently as attenuation increases.
for strength, trial in sorted(trials.items()):
    scores = trial["output"]["scores_edited"]
    ranks, base_ranks = [], []
    for ix in groups_d:
        g = ix[dgold[ix]]
        ranks.append(max(1 + int((scores[ix] > scores[j]).sum()) for j in g))
        base_ranks.append(max(1 + int((dscores[ix] > dscores[j]).sum()) for j in g))
    print("Strength", strength,
          "mean worst-gold rank:", round(float(np.mean(ranks)), 3),
          "baseline:", round(float(np.mean(base_ranks)), 3),
          "score/overlap correlation:", round(float(correlations(
              centered(scores, groups_d), centered(doverlap, groups_d))), 4))
    for typ in ["bridge", "comparison"]:
        mask = np.array([practice[int(q)]["type"] == typ for q in discovery_ids])
        print(" ", typ, "n", int(mask.sum()),
              "both@3", round(float(trial["metrics"]["success"][mask].mean()), 3),
              "baseline", round(float(base_d["success"][mask].mean()), 3))
    candidate_copy = target_original.clone()
    candidate_copy[:, target_columns] = target_original[:, target_columns] * (1.0 - strength)
    torch.save({target_name: candidate_copy}, WORK / ("candidate_alpha_" + str(strength) + ".pt"))
np.savez(WORK / "discovery_specification.npz",
         discovery_ids=discovery_ids, confirmation_ids=confirmation_ids, target_neurons=target_neurons)
print("Candidate copies and exact split/column specification saved under WORK.")
