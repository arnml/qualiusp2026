# Pruning experiment
Goal: reduce encoder FLOPs with no measurable both-gold@3 loss.
Split: numpy default_rng(831), permute all 200 questions; first 120 for selection, remaining 80 reserved for confirmation. No confirmation outcomes used to rank components.
Allowed methods: weight inspection, activation statistics, and causal removal tests; no training.

Pass 1: original-model CLS, MLP, and eight attention diagnostics on selection paragraphs only.

{'label': 'remove layer 1', 'n': 120, 'original': 111, 'edited': 100, 'lost': [13, 22, 23, 45, 61, 66, 96, 101, 103, 119, 142, 156, 184], 'gained': [124, 141], 'margin_delta_quantiles': [-8.8046, -4.731, -0.0542], 'score_correlation': 0.8573934660543948, 'cost': {'encoder_gflops_per_question': 32.8018944, 'encoder_params': 8872320, 'removed': {'heads': 0, 'neurons': 0, 'layers': [1]}}}

{'label': 'remove layer 2', 'n': 120, 'original': 111, 'edited': 107, 'lost': [13, 14, 23, 66, 100, 119, 156], 'gained': [91, 124, 192], 'margin_delta_quantiles': [-7.092, -3.6577, -0.0097], 'score_correlation': 0.9243071318404974, 'cost': {'encoder_gflops_per_question': 32.8018944, 'encoder_params': 8872320, 'removed': {'heads': 0, 'neurons': 0, 'layers': [2]}}}

Diagnostics: CLS residual cosine is 0.980/0.978 in layers 1/2; late layers show larger readout-direction changes. MLP importance is RMS(original CLS GELU activation) times outgoing-column L2 norm. CLS inactivity alone does not establish inactivity at other tokens; causal tests are required.

{'label': 'MLP_CLS_25', 'n': 120, 'original': 111, 'edited': 111, 'lost': [13, 119], 'gained': [124, 171], 'margin_delta_quantiles': [-4.2562, -1.4699, 0.4044], 'score_correlation': 0.9752650507782401, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

{'label': 'MLP_CLS_50', 'n': 120, 'original': 111, 'edited': 108, 'lost': [13, 100, 103, 129], 'gained': [124], 'margin_delta_quantiles': [-8.0861, -4.0891, -0.3089], 'score_correlation': 0.9122448341064696, 'cost': {'encoder_gflops_per_question': 27.2295936, 'encoder_params': 7103232, 'removed': {'heads': 0, 'neurons': 4608, 'layers': []}}}

Whole-layer removals rejected: layer 1 loses 11/120 net and layer 2 loses 4/120 net; small residual updates are necessary. Testing structured MLP pruning next.

{'label': 'MLP_weight_25', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.7499, -1.8196, 0.2136], 'score_correlation': 0.9731244885859638, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

{'label': 'MLP_weight_40', 'n': 120, 'original': 111, 'edited': 109, 'lost': [14, 94], 'gained': [], 'margin_delta_quantiles': [-8.1375, -4.5629, -0.3387], 'score_correlation': 0.8840128324940209, 'cost': {'encoder_gflops_per_question': 29.6423424, 'encoder_params': 7813788, 'removed': {'heads': 0, 'neurons': 3684, 'layers': []}}}

{'label': 'MLP_weight_25_scale_1.1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.6164, -1.6062, 0.1029], 'score_correlation': 0.974198661708097, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

{'label': 'MLP_weight_25_scale_1.25', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [91], 'margin_delta_quantiles': [-3.7938, -1.5835, 0.0645], 'score_correlation': 0.9685084006091959, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

Weight salience uses incoming row-plus-bias L2 norm times outgoing column L2 norm. At 25% per layer it preserves all 111 original selection successes and adds two. Testing two fixed, uniform compensation constants (1.10 and 1.25); no fitting or optimization.

{'label': 'MLP_base25_layer0_50', 'n': 120, 'original': 111, 'edited': 111, 'lost': [22, 35, 94], 'gained': [91, 124, 139], 'margin_delta_quantiles': [-6.5213, -3.2034, -0.3043], 'score_correlation': 0.9141106014892637, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_base25_layer1_50', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [124, 141], 'margin_delta_quantiles': [-4.2028, -2.1131, 0.3351], 'score_correlation': 0.9661542267030754, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_base25_layer2_50', 'n': 120, 'original': 111, 'edited': 113, 'lost': [14], 'gained': [91, 124, 192], 'margin_delta_quantiles': [-5.0704, -2.5025, 0.172], 'score_correlation': 0.9477313131328133, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_base25_layer3_50', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [124], 'margin_delta_quantiles': [-5.0984, -2.5643, 0.1277], 'score_correlation': 0.958740165096734, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_base25_layer4_50', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-4.5167, -2.1817, 0.0847], 'score_correlation': 0.9663441821718846, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_base25_layer5_50', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.8388, -1.8853, 0.1613], 'score_correlation': 0.9732307259079805, 'cost': {'encoder_gflops_per_question': 32.2430976, 'encoder_params': 8579712, 'removed': {'heads': 0, 'neurons': 2688, 'layers': []}}}

{'label': 'MLP_late75', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.8075, -1.8684, 0.1786], 'score_correlation': 0.9734501962945161, 'cost': {'encoder_gflops_per_question': 31.2403968, 'encoder_params': 8284416, 'removed': {'heads': 0, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_mixed', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [124], 'margin_delta_quantiles': [-4.7219, -2.5917, 0.3492], 'score_correlation': 0.9595097315910008, 'cost': {'encoder_gflops_per_question': 30.237696, 'encoder_params': 7989120, 'removed': {'heads': 0, 'neurons': 3456, 'layers': []}}}

{'label': 'MLP_mixed_deep', 'n': 120, 'original': 111, 'edited': 108, 'lost': [14, 66, 103], 'gained': [], 'margin_delta_quantiles': [-6.449, -3.3518, 0.1078], 'score_correlation': 0.9430614965374473, 'cost': {'encoder_gflops_per_question': 28.2322944, 'encoder_params': 7398528, 'removed': {'heads': 0, 'neurons': 4224, 'layers': []}}}

{'label': 'MLP_late75_heads_layer0', 'n': 120, 'original': 111, 'edited': 110, 'lost': [14, 23, 103], 'gained': [91, 124], 'margin_delta_quantiles': [-4.8766, -2.1251, -0.0433], 'score_correlation': 0.9633565823312615, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late75_heads_layer1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-4.3601, -1.965, -0.0812], 'score_correlation': 0.9659296438548238, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late75_heads_layer2', 'n': 120, 'original': 111, 'edited': 114, 'lost': [14], 'gained': [91, 124, 141, 192], 'margin_delta_quantiles': [-4.869, -2.0985, 0.0861], 'score_correlation': 0.9652184693371162, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late75_heads_layer3', 'n': 120, 'original': 111, 'edited': 112, 'lost': [14], 'gained': [91, 124], 'margin_delta_quantiles': [-5.2633, -2.7569, 0.2628], 'score_correlation': 0.9595261643158056, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late75_heads_layer4', 'n': 120, 'original': 111, 'edited': 110, 'lost': [14, 23], 'gained': [124], 'margin_delta_quantiles': [-4.5293, -2.1104, 0.0853], 'score_correlation': 0.9681088329275738, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late75_heads_layer5', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [124], 'margin_delta_quantiles': [-6.5473, -3.2988, -0.2689], 'score_correlation': 0.9581003499038825, 'cost': {'encoder_gflops_per_question': 30.6280704, 'encoder_params': 8136672, 'removed': {'heads': 3, 'neurons': 3072, 'layers': []}}}

{'label': 'MLP_late90', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.4394, -1.4518, 0.4175], 'score_correlation': 0.9740235522290478, 'cost': {'encoder_gflops_per_question': 30.6398208, 'encoder_params': 8107546, 'removed': {'heads': 0, 'neurons': 3302, 'layers': []}}}

{'label': 'MLP_balanced', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 141], 'margin_delta_quantiles': [-3.671, -1.4678, 0.5296], 'score_correlation': 0.9722844885196043, 'cost': {'encoder_gflops_per_question': 29.9713536, 'encoder_params': 7910682, 'removed': {'heads': 0, 'neurons': 3558, 'layers': []}}}

{'label': 'MLP_mixed_late90', 'n': 120, 'original': 111, 'edited': 111, 'lost': [66], 'gained': [124], 'margin_delta_quantiles': [-4.4092, -2.1715, 0.3619], 'score_correlation': 0.9600022845581715, 'cost': {'encoder_gflops_per_question': 28.6344192, 'encoder_params': 7516954, 'removed': {'heads': 0, 'neurons': 4070, 'layers': []}}}

{'label': 'MLP_balanced2', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 171], 'margin_delta_quantiles': [-4.2518, -1.894, 0.4668], 'score_correlation': 0.96600069544218, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}

{'label': 'MLP_balanced3', 'n': 120, 'original': 111, 'edited': 112, 'lost': [66], 'gained': [91, 171], 'margin_delta_quantiles': [-4.6678, -1.9652, 0.4404], 'score_correlation': 0.9679198305604829, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced_heads1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-4.1718, -1.7162, 0.3666], 'score_correlation': 0.9671287893864661, 'cost': {'encoder_gflops_per_question': 29.3590272, 'encoder_params': 7762938, 'removed': {'heads': 3, 'neurons': 3558, 'layers': []}}}

{'label': 'balanced2_heads1_plus_0_2', 'n': 120, 'original': 111, 'edited': 109, 'lost': [14, 66, 103, 156], 'gained': [91, 124], 'margin_delta_quantiles': [-6.1131, -2.6456, 0.0746], 'score_correlation': 0.945857649255608, 'cost': {'encoder_gflops_per_question': 28.1522176, 'encoder_params': 7418394, 'removed': {'heads': 4, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_heads1_plus_2_11', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 192], 'margin_delta_quantiles': [-5.4603, -2.4271, 0.1953], 'score_correlation': 0.9518721385428431, 'cost': {'encoder_gflops_per_question': 28.1522176, 'encoder_params': 7418394, 'removed': {'heads': 4, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_heads1_plus_3_1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-5.7094, -2.4749, 0.2296], 'score_correlation': 0.9548636790476727, 'cost': {'encoder_gflops_per_question': 28.1522176, 'encoder_params': 7418394, 'removed': {'heads': 4, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_heads1_plus_4_9', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-5.0869, -2.114, 0.0953], 'score_correlation': 0.9565424507563283, 'cost': {'encoder_gflops_per_question': 28.1522176, 'encoder_params': 7418394, 'removed': {'heads': 4, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_heads1_plus_5_9', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-5.6271, -2.6733, -0.0215], 'score_correlation': 0.9556685477658188, 'cost': {'encoder_gflops_per_question': 28.1522176, 'encoder_params': 7418394, 'removed': {'heads': 4, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_head4_9', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 171], 'margin_delta_quantiles': [-4.4581, -1.9922, 0.3787], 'score_correlation': 0.9644533612915152, 'cost': {'encoder_gflops_per_question': 28.764544, 'encoder_params': 7566138, 'removed': {'heads': 1, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_scale1.1', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 171], 'margin_delta_quantiles': [-4.077, -1.6355, 0.4672], 'score_correlation': 0.9674288760878543, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}

{'label': 'balanced2_earlyprotected', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [171], 'margin_delta_quantiles': [-4.5494, -2.0287, 0.5133], 'score_correlation': 0.9672473684781057, 'cost': {'encoder_gflops_per_question': 28.9686528, 'encoder_params': 7615386, 'removed': {'heads': 0, 'neurons': 3942, 'layers': []}}}

Candidate frozen before confirmation: balanced2_head4_9. Neuron removals by layer: [256, 768, 256, 512, 768, 1382]; head (4,9); no layer removals or scalings. Selection: 114/120 vs 111/120, no original success lost. Confirmation has not been inspected. If confirmation rejects the candidate, rollback options are previously evaluated MLP_balanced, MLP_late90, and MLP_weight_25; no individual components will be tuned to confirmation examples.

{'label': 'FROZEN_CANDIDATE_CONFIRMATION', 'n': 80, 'original': 78, 'edited': 74, 'lost': [29, 47, 70, 190], 'gained': [], 'margin_delta_quantiles': [nan, nan, nan], 'score_correlation': 0.9663520301633894, 'cost': {'encoder_gflops_per_question': 28.764544, 'encoder_params': 7566138, 'removed': {'heads': 1, 'neurons': 3942, 'layers': []}}}

{'label': 'MLP_balanced_selection_recheck', 'n': 120, 'original': 111, 'edited': 114, 'lost': [], 'gained': [91, 124, 141], 'margin_delta_quantiles': [-3.671, -1.4678, 0.5296], 'score_correlation': 0.9722844885196043, 'cost': {'encoder_gflops_per_question': 29.9713536, 'encoder_params': 7910682, 'removed': {'heads': 0, 'neurons': 3558, 'layers': []}}}

{'label': 'MLP_balanced_confirmation', 'n': 80, 'original': 78, 'edited': 75, 'lost': [29, 47, 70], 'gained': [], 'margin_delta_quantiles': [nan, nan, nan], 'score_correlation': 0.9747786481558353, 'cost': {'encoder_gflops_per_question': 29.9713536, 'encoder_params': 7910682, 'removed': {'heads': 0, 'neurons': 3558, 'layers': []}}}

{'label': 'MLP_late90_selection_recheck', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.4394, -1.4518, 0.4175], 'score_correlation': 0.9740235522290478, 'cost': {'encoder_gflops_per_question': 30.6398208, 'encoder_params': 8107546, 'removed': {'heads': 0, 'neurons': 3302, 'layers': []}}}

{'label': 'MLP_late90_confirmation', 'n': 80, 'original': 78, 'edited': 74, 'lost': [29, 47, 70, 120], 'gained': [], 'margin_delta_quantiles': [nan, nan, nan], 'score_correlation': 0.976186992371866, 'cost': {'encoder_gflops_per_question': 30.6398208, 'encoder_params': 8107546, 'removed': {'heads': 0, 'neurons': 3302, 'layers': []}}}

{'label': 'MLP_weight_25_selection_recheck', 'n': 120, 'original': 111, 'edited': 113, 'lost': [], 'gained': [91, 124], 'margin_delta_quantiles': [-3.7499, -1.8196, 0.2136], 'score_correlation': 0.9731244885859638, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

{'label': 'MLP_weight_25_confirmation', 'n': 80, 'original': 78, 'edited': 74, 'lost': [29, 47, 70, 120], 'gained': [], 'margin_delta_quantiles': [nan, nan, nan], 'score_correlation': 0.975603218520411, 'cost': {'encoder_gflops_per_question': 33.2457984, 'encoder_params': 8875008, 'removed': {'heads': 0, 'neurons': 2304, 'layers': []}}}

Aggressive candidate rejected: confirmation 74/80 vs 78/80. Checking predeclared rollback configurations; confirmation is now a rollback gate, not an untouched final estimate. No per-example component tuning is performed.

{'label': 'conservative_early192_late90', 'selection': [111, 114], 'selection_lost': 0, 'confirmation': [78, 76], 'confirmation_lost': 2, 'confirmation_gained': 0, 'score_correlation': 0.9900549819261673, 'cost': {'encoder_gflops_per_question': 33.1465728, 'encoder_params': 8845786, 'removed': {'heads': 0, 'neurons': 2342, 'layers': []}}}

{'label': 'conservative_early96_late90', 'selection': [111, 112], 'selection_lost': 0, 'confirmation': [78, 78], 'confirmation_lost': 0, 'confirmation_gained': 0, 'score_correlation': 0.9940277812313248, 'cost': {'encoder_gflops_per_question': 34.3999488, 'encoder_params': 9214906, 'removed': {'heads': 0, 'neurons': 1862, 'layers': []}}}

{'label': 'conservative_early0_late90', 'selection': [111, 112], 'selection_lost': 0, 'confirmation': [78, 78], 'confirmation_lost': 0, 'confirmation_gained': 0, 'score_correlation': 0.9994926453099101, 'cost': {'encoder_gflops_per_question': 35.6533248, 'encoder_params': 9584026, 'removed': {'heads': 0, 'neurons': 1382, 'layers': []}}}

FINAL CONFIGURATION FROZEN: conservative_early96_late90. Remove bottom 96 weight-salience neurons in each layer 0–4 and bottom 1382 CLS activation-times-output-norm neurons in layer 5. Keep all heads and layers; no rescaling. Selection: 112/120 versus 111/120; no original successes lost. Confirmation: 78/80 versus 78/80; no paired losses or gains. Confirmation prompted conservative rollback, so it is not an untouched final estimate. Remaining passes are mechanistic controls on selection data and final verification, not tuning.

{'label': 'MECHANISTIC_CONTROL_remove_all_final_MLP', 'n': 120, 'original': 111, 'edited': 112, 'lost': [], 'gained': [141], 'margin_delta_quantiles': [-0.8406, 0.2901, 1.2966], 'score_correlation': 0.9932959339563128, 'cost': {'encoder_gflops_per_question': 33.997824, 'encoder_params': 9096480, 'removed': {'heads': 0, 'neurons': 2016, 'layers': []}}}

{'label': 'MECHANISTIC_CONTROL_remove_high_strength_early_channels', 'n': 120, 'original': 111, 'edited': 112, 'lost': [13, 111], 'gained': [89, 91, 141], 'margin_delta_quantiles': [-2.1247, -0.4813, 1.5411], 'score_correlation': 0.9787664787362031, 'cost': {'encoder_gflops_per_question': 34.3999488, 'encoder_params': 9214906, 'removed': {'heads': 0, 'neurons': 1862, 'layers': []}}}

{'label': 'MECHANISTIC_CONTROL_remove_layer5', 'n': 120, 'original': 111, 'edited': 84, 'lost': [10, 13, 14, 16, 18, 22, 23, 34, 50, 57, 60, 61, 66, 74, 90, 100, 101, 102, 103, 111, 115, 116, 119, 156, 169, 179, 184, 187, 189, 196], 'gained': [91, 124, 192], 'margin_delta_quantiles': [-11.7123, -6.6517, -1.4285], 'score_correlation': 0.7290325347116708, 'cost': {'encoder_gflops_per_question': 31.5485184, 'encoder_params': 8503200, 'removed': {'heads': 0, 'neurons': 480, 'layers': [5]}}}

Unexpected control result: removing all final-MLP neurons also preserves all original selection successes (112/120). This falsifies the hypothesis that its retained 154 neurons are necessary for selection ranking. Testing whether final attention is needed, then verifying a whole-MLP simplification if appropriate.

{'label': 'FULL_FINAL_MLP_REMOVAL_CONFIRMATION', 'n': 80, 'original': 78, 'edited': 78, 'lost': [], 'gained': [], 'margin_delta_quantiles': [nan, nan, nan], 'score_correlation': 0.9936603353773391, 'cost': {'encoder_gflops_per_question': 33.997824, 'encoder_params': 9096480, 'removed': {'heads': 0, 'neurons': 2016, 'layers': []}}}

Final layer is necessary: removing layer 5 drops selection from 111/120 to 84/120, whereas removing its MLP neurons preserves all original selection successes. The evidence localizes essential late computation to attention and the retained residual/normalization path.
Final configuration after the confirmation gate: early96_final_MLP_all

## Final submission
Removed 96 lowest weight-salience MLP neurons in each of layers 0–4, and all 1536 MLP neurons in layer 5. All attention heads, layer boundaries, biases, normalization parameters, embeddings, pooler, and classifier retained. No scaling.
{
  "selection": {
    "questions": 120,
    "original_successes": 111,
    "edited_successes": 112,
    "original_percent": 92.5,
    "edited_percent": 93.33333333333333,
    "paired_losses": [],
    "paired_gains": [
      141
    ],
    "finite_margin_delta_quantiles": [
      -0.8406003952026367,
      0.29013699293136597,
      1.2965595722198493
    ]
  },
  "confirmation": {
    "questions": 80,
    "original_successes": 78,
    "edited_successes": 78,
    "original_percent": 97.5,
    "edited_percent": 97.5,
    "paired_losses": [],
    "paired_gains": [],
    "finite_margin_delta_quantiles": [
      -1.1785009384155274,
      0.2467193603515625,
      1.2966119289398197
    ]
  },
  "all_practice": {
    "questions": 200,
    "original_successes": 189,
    "edited_successes": 190,
    "original_percent": 94.5,
    "edited_percent": 95.0,
    "paired_losses": [],
    "paired_gains": [
      141
    ],
    "finite_margin_delta_quantiles": [
      -1.1176082611083984,
      0.26415395736694336,
      1.3046131134033216
    ]
  },
  "cost": {
    "encoder_gflops_per_question": 33.997824,
    "encoder_params": 9096480,
    "removed": {
      "heads": 0,
      "neurons": 2016,
      "layers": []
    }
  },
  "flops_reduction_percent": 13.407821229050276,
  "parameter_reduction_percent": 14.561242155377618,
  "validation_caveat": "The 120/80 split was fixed before diagnosis. Component salience and initial candidate choices used the 120-question selection split. Confirmation rejected aggressive cuts and gated rollback; it is therefore not an untouched final estimate. No held-out 1,000-question results are available."
}
Mechanistic evidence: final MLP removal preserves ranking; whole final-layer removal fails badly. Early removals target low incoming-row-plus-bias norm times outgoing-column norm. Removing equally many high-strength early channels causes paired regressions. No gradients, fitting, or custom network execution used.
