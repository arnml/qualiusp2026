OUTPUT:
```
[forward] pass used; 0 left
selection {'questions': 120, 'original_successes': 111, 'edited_successes': 112, 'original_percent': 92.5, 'edited_percent': 93.33333333333333, 'paired_losses': [], 'paired_gains': [141], 'finite_margin_delta_quantiles': [-0.8406003952026367, 0.29013699293136597, 1.2965595722198493]}
confirmation {'questions': 80, 'original_successes': 78, 'edited_successes': 78, 'original_percent': 97.5, 'edited_percent': 97.5, 'paired_losses': [], 'paired_gains': [], 'finite_margin_delta_quantiles': [-1.1785009384155274, 0.2467193603515625, 1.2966119289398197]}
all_practice {'questions': 200, 'original_successes': 189, 'edited_successes': 190, 'original_percent': 94.5, 'edited_percent': 95.0, 'paired_losses': [], 'paired_gains': [141], 'finite_margin_delta_quantiles': [-1.1176082611083984, 0.26415395736694336, 1.3046131134033216]}
Final cost: {'encoder_gflops_per_question': 33.997824, 'encoder_params': 9096480, 'removed': {'heads': 0, 'neurons': 2016, 'layers': []}}
FLOPs reduction: 13.408 %
Final state: {'track': 'C', 'heads_removed': [], 'neurons_removed': 2016, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 33.997824, 'encoder_params': 9096480, 'removed': {'heads': 0, 'neurons': 2016, 'layers': []}}}
Notes: C:\code\qualiusp2026\experiments\exp04_astra_edit\runs\s3_C_B50\work\notes.md
Manifest: C:\code\qualiusp2026\experiments\exp04_astra_edit\runs\s3_C_B50\work\pruning_manifest.json

```
Passes left: 0/50. Turn 22/200. The forward-pass budget is used up. Code still runs (forward() is refused). Make sure your current removals are your final edit, then reply WITHOUT a code block with your final explanation (mechanism, changes, evidence).