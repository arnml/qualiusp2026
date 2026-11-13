# Weight-edit experiment
Fixed random split, seed 20261113: 100 discovery questions and 100 confirmation questions (or equal halves if the supplied count differs). Confirmation outcomes will remain unexamined until the candidate and its strength are fixed.
Plan: baseline with representations; three discovery interventions; final paired confirmation.
No optimization, fitted predictive models, or private model execution. Only descriptive statistics, weight inspection, and the provided forward function.

Baseline discovery: {'both@3': 0.55, 'tokens@90': 950.47, 'retention': 0.9, 'cutoff': -6.597182750701904}

Mechanism candidate: layer 4 MLP outputs promoting lexical distractors. Screened using within-question activation correlations, distractor-minus-weak-gold means, and output-weight alignment with the layer's score-associated CLS direction. These are descriptive moments, not a fitted objective.
Fixed selected columns: [188, 1202, 75, 373, 869, 1503, 1481, 1270, 425, 299, 981, 314, 1106, 635, 511, 389, 120, 380, 281, 25, 1238, 1368, 691, 1291, 1357, 525, 1429, 698, 1278, 212, 358, 767, 1109, 1279, 1318, 319, 1426, 110, 440, 408, 1425, 1329, 135, 1331, 280, 752, 1366, 1170, 567, 890, 911, 529, 1310, 201, 805, 292, 22, 1379, 327, 770, 564, 1224, 460, 1113, 976, 1251, 720, 435, 1086, 960, 1499, 610, 994, 395, 1466, 3, 260, 1234, 1286, 1356, 1369, 747, 798, 63, 335, 750, 1084, 1519, 1073, 1028, 422, 924, 613, 279, 49, 1022]
Intervention W[:, S] = (1-alpha) W_original[:, S]; 36,864 affected entries.
Strength 0.5: {'both@3': 0.55, 'tokens@90': 913.2, 'top3_gain_pp': 0.0, 'paired_top3_se_pp': 0.0, 'wins': 0, 'losses': 0, 'weak_gold_vs_second_distractor_margin_change': 0.03865785151720047}
Strength 2.0: {'both@3': 0.55, 'tokens@90': 920.19, 'top3_gain_pp': 0.0, 'paired_top3_se_pp': 0.0, 'wins': 0, 'losses': 0, 'weak_gold_vs_second_distractor_margin_change': 0.058399878442287445}
Strength 1.0: {'both@3': 0.55, 'tokens@90': 911.52, 'top3_gain_pp': 0.0, 'paired_top3_se_pp': 0.0, 'wins': 0, 'losses': 0, 'weak_gold_vs_second_distractor_margin_change': 0.06897977739572525}
Caution: stronger attenuation did not monotonically improve top3 or token retention. Observed score movement alone is not evidence of ranking repair.

Predeclared confirmation rule: Freeze the strength with lowest discovery tokens@90 (ties favor smaller strength), then evaluate it once on confirmation questions. Keep only if confirmation token saving exceeds twice its paired bootstrap standard error, its 95% bootstrap interval excludes zero, and both-gold@3 does not show a deterioration exceeding twice its paired standard error. Otherwise restore the original tensor.
Token-cost estimates use supplied tokenizer paragraph lengths, without prompt wrappers.

Frozen candidate before confirmation: alpha=1.0.
Confirmation: {'strength': 1.0, 'questions': 100, 'baseline_both@3': 0.54, 'edited_both@3': 0.55, 'wins': 1, 'losses': 0, 'baseline_tokens@90': 962.67, 'edited_tokens@90': 936.71, 'token_saving': 25.959999999999923, 'token_saving_percent': 2.696666562788902, 'token_saving_se': 15.897326328634756, 'token_saving_95pct_interval': [-18.311249999999944, 49.49075], 'top3_gain_se_pp': 1.0145963340877142, 'top3_gain_95pct_interval_pp': [0.0, 3.0000000000000027]}
Meets predeclared retention rule: False

Final action: Restored the original weights because the confirmation criterion was not met.
Final changed entries in tested tensor: 0
Final edited(): {'track': 'R', 'heads_removed': [], 'neurons_removed': 0, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}}
Interpretation: discovery showed an overlap-associated signal, but causal ranking gains must be distinguished from score shifts and sampling-sensitive threshold changes. The final submission follows the untouched confirmation result.

Final verification: original weights restored; no candidate retained. Confirmation saving was 25.96 tokenizer tokens/question, SE 15.90, 95% paired bootstrap interval [-18.31, 49.49]. Both-gold@3 changed from 54% to 55% (one win, zero losses).
