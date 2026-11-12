# Pruning notes
Goal: reduce encoder FLOPs while preserving both-gold@3 within one percentage point.
No training, gradients, fitted objectives, or independent model execution.
Use a deterministic 100-question selection / 100-question confirmation split; reserve confirmation outcomes until a candidate is chosen.

Split seed: 20261112. Selection indices: [0, 2, 3, 4, 13, 14, 20, 21, 22, 23, 26, 27, 28, 29, 33, 36, 41, 42, 46, 47, 49, 52, 55, 56, 60, 61, 62, 63, 67, 68, 69, 74, 75, 76, 77, 79, 80, 82, 86, 87, 89, 90, 91, 92, 93, 94, 95, 99, 102, 103, 105, 108, 109, 110, 111, 115, 117, 121, 123, 124, 125, 126, 129, 130, 131, 132, 134, 135, 138, 139, 142, 146, 147, 148, 150, 156, 158, 159, 162, 163, 164, 166, 167, 168, 170, 171, 173, 177, 178, 179, 180, 182, 183, 185, 186, 189, 191, 192, 196, 199]
Static diagnostics: input/output norms and composed attention value/output weight norms. These are weight statistics, not an independently executed forward pass.

Pass 1: original model on the selection half only; CLS, MLP activations, and attention for eight examples requested. Confirmation remains unexamined.

Selection baseline successes: 92/100.
Statistics: activation RMS times MLP output-column norm; CLS interlayer cosine and relative change; correlation of CLS changes projected on the static pooler/classifier weight direction with scores; structural-token attention and question-to-paragraph attention. No fitted objectives.

Candidate A: MLP removals per layer [384,512,512,512,640,1024]; no heads/layers removed.
Ranking for layers 0-4: output-column norm * input-row norm * sqrt(0.25 + CLS RMS/median RMS). Final-layer ranking: output-column norm * CLS RMS. All statistics use selection only.
Pass 2 paired selection result: {'n': 100, 'original_successes': 92, 'edited_successes': 93, 'losses': [47], 'gains': [124, 139], 'margin_change_q10_median_q90': [-4.8861, -2.4556, 0.3293], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 71}, 'comparison': {'n': 22, 'original': 22, 'edited': 22}}}
Cost: {'encoder_gflops_per_question': 29.9034624, 'encoder_params': 7890688, 'removed': {'heads': 0, 'neurons': 3584, 'layers': []}}

Candidate A saves 23.84% FLOPs, with one selection regression and two gains. Whole layers retained because CLS similarity alone does not establish token-level redundancy.
Attention candidates examined using both directions of cross-segment attention, CLS targets, weight-path norms, and static readout projection. Small attention sample is hypothesis-generating only.

Candidate B adds heads [(0, 2), (1, 6), (1, 7), (1, 9), (2, 1), (2, 11), (3, 1), (4, 9)] to candidate A.
Evidence: layer 0 head 2 attends exclusively to CLS from CLS; layer 1 candidates predominantly attend to CLS there and have low cross-segment attention; layer 2/3 candidates combine weak weight paths with little cross-segment attention; layer 4 head 9 has the weakest value/output path in its layer. Later lexical question-attending heads were retained. These are hypotheses, not claims of zero contribution.
Pass 3 paired selection result: {'n': 100, 'original_successes': 92, 'edited_successes': 88, 'losses': [14, 29, 47, 86, 103, 156], 'gains': [124, 192], 'margin_change_q10_median_q90': [-7.5442, -2.9288, -0.0076], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 67}, 'comparison': {'n': 22, 'original': 22, 'edited': 21}}}
Cost: {'encoder_gflops_per_question': 28.270592, 'encoder_params': 7496704, 'removed': {'heads': 8, 'neurons': 3584, 'layers': []}}

Candidate B rejected: 88/100 versus 92/100 original; all heads restored.
Candidate C MLP removals [640,768,768,512,640,1152], no heads or layers removed. Incremental pruning emphasizes earlier layers with very small observed CLS drift under A; this remains a hypothesis because other-token activations are unavailable.
Pass 4 selection result: {'n': 100, 'original_successes': 92, 'edited_successes': 88, 'losses': [29, 47, 76, 103, 129], 'gains': [124], 'margin_change_q10_median_q90': [-7.7878, -3.9825, -0.022], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 66}, 'comparison': {'n': 22, 'original': 22, 'edited': 22}}}
Within-question score correlation: 0.9368991496723466
Selection-only preferred candidate, fixed before confirmation inspection: A_mlp
Confirmation arrays collected but no confirmation statistics inspected.

Candidate C rejected on selection alone: 88/100 and score correlation 0.9369. Its confirmation outcomes were not inspected or used.
Final chosen candidate A restored before pass 5: 3584 neurons, counts [384,512,512,512,640,1024]; all heads/layers retained; no rescalings.
Pass 5 selection: {'n': 100, 'original_successes': 92, 'edited_successes': 93, 'losses': [47], 'gains': [124, 139], 'margin_change_q10_median_q90': [-4.8861, -2.4556, 0.3293], 'by_type': {'bridge': {'n': 78, 'original': 70, 'edited': 71}, 'comparison': {'n': 22, 'original': 22, 'edited': 22}}}
Pass 5 previously untouched confirmation: {'n': 100, 'original_successes': 97, 'edited_successes': 96, 'losses': [70], 'gains': [], 'margin_change_q10_median_q90': [-4.5597, -2.5129, -0.3693], 'by_type': {'bridge': {'n': 85, 'original': 82, 'edited': 81}, 'comparison': {'n': 15, 'original': 15, 'edited': 15}}}
Pass 5 all practice: {'n': 200, 'original_successes': 189, 'edited_successes': 189, 'losses': [47, 70], 'gains': [124, 139], 'margin_change_q10_median_q90': [-4.8265, -2.4764, 0.0495], 'by_type': {'bridge': {'n': 163, 'original': 152, 'edited': 152}, 'comparison': {'n': 37, 'original': 37, 'edited': 37}}}
Final cost: {'encoder_gflops_per_question': 29.9034624, 'encoder_params': 7890688, 'removed': {'heads': 0, 'neurons': 3584, 'layers': []}}
FLOP reduction: 23.836126629422726%.
Exact neuron indices saved in /work/removal_plan.json. CPU timing was not measured here.
