# Hypothesis: the discovery-selected removal of 96 layer-4 MLP outputs yields a reproducible reduction in token retention.
chosen_strength = min(trials, key=lambda a: (trials[a]["metrics"]["tokens@90"], a))
assert chosen_strength == 1.0
install_attenuation(chosen_strength)
torch.save({target_name: load(target_name)}, WORK / "confirmation_candidate.pt")
with NOTES.open("a") as f:
    f.write("\nFrozen candidate before confirmation: alpha=" + str(chosen_strength) + ".\n")

confirmation = forward(items=confirmation_items, model="both", cls=True)
cbase = metrics(confirmation["scores_original"], groups_c, cgold, clength)
cedit = metrics(confirmation["scores_edited"], groups_c, cgold, clength)
cuncertainty = paired_bootstrap(
    confirmation["scores_original"], confirmation["scores_edited"],
    groups_c, cgold, clength, repetitions=5000, seed=92731
)
ctop_delta = cedit["success"].astype(float) - cbase["success"]
ctop_se = float(ctop_delta.std(ddof=1) / np.sqrt(len(ctop_delta)))
ctoken_saving = float(cbase["tokens@90"] - cedit["tokens@90"])
confirmation_report = {
    "strength": chosen_strength,
    "questions": len(groups_c),
    "baseline_both@3": cbase["both@3"],
    "edited_both@3": cedit["both@3"],
    "wins": int((ctop_delta > 0).sum()),
    "losses": int((ctop_delta < 0).sum()),
    "baseline_tokens@90": cbase["tokens@90"],
    "edited_tokens@90": cedit["tokens@90"],
    "token_saving": ctoken_saving,
    "token_saving_percent": 100 * ctoken_saving / cbase["tokens@90"],
    **cuncertainty
}
keep_candidate = bool(
    ctoken_saving > 2 * cuncertainty["token_saving_se"]
    and cuncertainty["token_saving_95pct_interval"][0] > 0
    and ctop_delta.mean() >= -2 * ctop_se
)
print("Untouched confirmation:", confirmation_report)
print("Meets predeclared retention rule:", keep_candidate)
with NOTES.open("a") as f:
    f.write("Confirmation: " + repr(confirmation_report) + "\n")
    f.write("Meets predeclared retention rule: " + str(keep_candidate) + "\n")
