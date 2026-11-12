# Pruning experiment

Goal: reduce encoder FLOPs while preserving both-gold@3 within one percentage point.
Split: seeded permutation (20261112), 120 discovery questions and 80 confirmation questions. Confirmation questions will not guide removal selection.
Methods permitted and planned: weight statistics, activation statistics, attention inspection, and paired causal removal experiments; no training, gradients, or fitted objectives.
Forward budget: 10 calls. Inspect weights before choosing diagnostic probes.

Weight inspection: no exactly zero tensors or obvious deleted components. Pass 1 measures all discovery paragraphs, with CLS, MLP, and eight attention probes. Probes pair weak lexical-overlap gold paragraphs with strong-overlap distractors.

Discovery baseline both-gold@3: 111/120.
Diagnostics: MLP activation RMS times output-column norm; input/output weight norms; gold–distractor activation contrast; CLS layer cosine; head value/output norm product; structural-token and paragraph attention on eight probes. Pooler/classifier weight product is a static direction proxy, not a simulated forward or gradient.

{"name": "drop_layer1", "original": 111, "edited": 99, "n": 120, "losses": [13, 22, 23, 45, 47, 61, 68, 86, 103, 119, 135, 142, 143, 156], "gains": [124, 141], "margin_delta_quantiles": [-11.037, -7.217, -3.782, -1.795, 0.904], "score_correlation": 0.85496, "cost": {"encoder_gflops_per_question": 32.8018944, "encoder_params": 8872320, "removed": {"heads": 0, "neurons": 0, "layers": [1]}}}

Layer 1 ablation rejected: 14 paired losses and 2 gains; all layers restored. Small CLS change did not imply dispensability, consistent with useful token processing.
MLP A uses low input/output weight magnitude combined with low CLS output-energy proxies. Earlier layers emphasize weight magnitude to protect neurons active away from CLS. Large gold–distractor activation contrasts receive additional protection. Last-layer pruning emphasizes CLS energy and contrast because no later token mixing occurs. No compensating scaling is applied.

{"name": "mlp_A", "original": 111, "edited": 112, "n": 120, "losses": [47], "gains": [124, 139], "margin_delta_quantiles": [-5.142, -3.149, -1.783, -0.602, 0.949], "score_correlation": 0.97631, "cost": {"encoder_gflops_per_question": 30.237696, "encoder_params": 7989120, "removed": {"heads": 0, "neurons": 3456, "layers": []}}}

MLP A retained as an eligible discovery candidate: 112/120 versus 111/120, one loss and two gains, encoder 30.237696 GFLOPs/question. Margins shrank, so score correlation and paired transitions remain selection checks. Additional head diagnostics use static O_h V_h Frobenius/readout norms and cosine similarity of observed CLS-attention profiles, with no network reconstruction.

Head candidate (zero-based): L0H2 has weak weights and CLS attention almost identical to a retained structural head; L1H7/H8 have weak static readout norms and highly redundant CLS attention; L2H11 and L3H1 have the weakest value/output maps in their layers; L4H9 has the weakest map and static readout norm in layer 4; L5H9 has the weakest map and overlaps H4's question-focused attention; L5H11 has the weakest static readout norm and much structural attention. These are hypotheses to test causally, not claims of proven redundancy.

{"name": "mlp_A_heads8", "original": 111, "edited": 102, "n": 120, "losses": [14, 23, 29, 47, 86, 103, 119, 142, 156], "gains": [], "margin_delta_quantiles": [-8.361, -5.416, -2.873, -1.085, 0.389], "score_correlation": 0.94743, "cost": {"encoder_gflops_per_question": 28.6048256, "encoder_params": 7595136, "removed": {"heads": 8, "neurons": 3456, "layers": []}}}

Eight-head combination rejected: 102/120, nine losses against the original. The extra savings did not justify its quality damage. Restore all heads, and test greater MLP sparsity using the same fixed diagnostic ranking. MLP B removal counts: [640,640,640,768,896,1280].

{"name": "mlp_B", "original": 111, "edited": 109, "n": 120, "losses": [47, 86, 103], "gains": [124], "margin_delta_quantiles": [-8.819, -5.779, -4.033, -1.601, 0.48], "score_correlation": 0.9272, "cost": {"encoder_gflops_per_question": 26.5611264, "encoder_params": 6906368, "removed": {"heads": 0, "neurons": 4864, "layers": []}}}

MLP B: 109/120 versus 111/120, three losses and one gain; unscaled version fails the discovery one-point requirement. Test a prespecified energy-compensation rule: sqrt(total proxy squared norm / retained proxy squared norm), capped at 1.25, using weight products in layers 0–3 and observed CLS energy in layers 4–5. The 16 highest-energy columns per layer are not scaled, to protect dominant channels. Factors are statistics, not fitted to scores, labels, or a loss.

{"name": "mlp_B_scaled", "original": 111, "edited": 109, "n": 120, "losses": [29, 47, 86, 103], "gains": [124, 139], "margin_delta_quantiles": [-8.888, -5.144, -3.648, -1.471, 0.718], "score_correlation": 0.93259, "cost": {"encoder_gflops_per_question": 26.5611264, "encoder_params": 6906368, "removed": {"heads": 0, "neurons": 4864, "layers": []}}}

Norm compensation did not recover discovery quality (109/120); discard all scaling. MLP C backs off pruning relative to B, particularly in layer 4, where the original CLS representation begins strong alignment with the static classifier direction. Counts [512,512,512,640,640,1280], no heads/layers removed and no scaling.

{"name": "mlp_C", "original": 111, "edited": 112, "n": 120, "losses": [47], "gains": [124, 192], "margin_delta_quantiles": [-6.966, -4.332, -2.72, -0.868, 0.848], "score_correlation": 0.95791, "cost": {"encoder_gflops_per_question": 28.566528, "encoder_params": 7496960, "removed": {"heads": 0, "neurons": 4096, "layers": []}}}

MLP C: 112/120, one loss and two gains, 28.566528 GFLOPs/question; eligible candidate. MLP D tests localization of B's damage: retain B's earlier pruning but restore layer 4 to the C/A count of 640 removed neurons. All rankings and importance measures stay fixed.

{"name": "mlp_D", "original": 111, "edited": 110, "n": 120, "losses": [47, 86, 103], "gains": [124, 192], "margin_delta_quantiles": [-8.308, -5.572, -3.863, -1.491, 0.557], "score_correlation": 0.93041, "cost": {"encoder_gflops_per_question": 27.2295936, "encoder_params": 7103232, "removed": {"heads": 0, "neurons": 4608, "layers": []}}}

MLP D: 110/120, three losses and two gains, 27.2295936 GFLOPs/question. Although its net decline is 0.83 points, its additional paired losses and weaker score correlation make C the more robust candidate. Final discovery experiment isolates L4H9 (weakest value/output map and readout norm in that layer) and L5H11 (weakest static readout norm), rather than the rejected eight-head set. Selection rule before confirmation: require at least the original 111/120 and at most one paired loss; among eligible candidates select lowest FLOPs, excluding whole-layer ablations. If the two-head version fails this rule, choose C. Confirmation labels/scores remain unused.

{"name": "mlp_C_heads2", "original": 111, "edited": 110, "n": 120, "losses": [23, 47], "gains": [124], "margin_delta_quantiles": [-7.15, -4.431, -2.902, -0.841, 0.79], "score_correlation": 0.95667, "cost": {"encoder_gflops_per_question": 28.1583104, "encoder_params": 7398464, "removed": {"heads": 2, "neurons": 4096, "layers": []}}}

Frozen selection before confirmation: mlp_C. No changes will be selected using the reserved confirmation results.

{"name": "confirmation_mlp_C", "original": 78, "edited": 77, "n": 80, "losses": [70], "gains": [], "margin_delta_quantiles": [-6.128, -4.018, -2.315, -1.457, -0.088], "score_correlation": 0.96146, "cost": {"encoder_gflops_per_question": 28.566528, "encoder_params": 7496960, "removed": {"heads": 0, "neurons": 4096, "layers": []}}}

## Final confirmation
{
  "chosen": "mlp_C",
  "discovery": {
    "name": "mlp_C",
    "original": 111,
    "edited": 112,
    "n": 120,
    "losses": [
      47
    ],
    "gains": [
      124,
      192
    ],
    "margin_delta_quantiles": [
      -6.966,
      -4.332,
      -2.72,
      -0.868,
      0.848
    ],
    "score_correlation": 0.95791,
    "cost": {
      "encoder_gflops_per_question": 28.566528,
      "encoder_params": 7496960,
      "removed": {
        "heads": 0,
        "neurons": 4096,
        "layers": []
      }
    }
  },
  "confirmation": {
    "name": "confirmation_mlp_C",
    "original": 78,
    "edited": 77,
    "n": 80,
    "losses": [
      70
    ],
    "gains": [],
    "margin_delta_quantiles": [
      -6.128,
      -4.018,
      -2.315,
      -1.457,
      -0.088
    ],
    "score_correlation": 0.96146,
    "cost": {
      "encoder_gflops_per_question": 28.566528,
      "encoder_params": 7496960,
      "removed": {
        "heads": 0,
        "neurons": 4096,
        "layers": []
      }
    }
  },
  "cost": {
    "encoder_gflops_per_question": 28.566528,
    "encoder_params": 7496960,
    "removed": {
      "heads": 0,
      "neurons": 4096,
      "layers": []
    }
  },
  "flops_reduction_percent": 27.24128757648311,
  "confirmation_delta_percentage_points": -1.25,
  "selection_used_confirmation": false
}
All ten forward passes used. Confirmation consisted of 80 previously unused questions evaluated through original and edited models on identical items. No final configuration changes were made using confirmation results. Finite practice samples do not establish a guaranteed held-out one-point bound. CPU timing is left to the physical-deletion evaluator.
