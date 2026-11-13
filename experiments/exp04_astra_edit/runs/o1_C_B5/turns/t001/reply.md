I'll start by inspecting the environment: tensor names, the cost baseline, the practice data, and the `forward` signature. This uses no passes.

```python
# Hypothesis: none yet. Orient myself (names, shapes, cost formula behaviour, data sizes) without spending passes.
import numpy as np, torch, inspect, json, collections
print(len(NAMES)); print([n for n in NAMES if 'layer.0.' in n or 'layer' not in n])
print(cost())
print(len(practice), practice[0].keys())
print(collections.Counter(len(q['paragraphs']) for q in practice))
print(collections.Counter(q['type'] for q in practice))
print(sum(len(q['paragraphs']) for q in practice), "total items")
try:
    print(inspect.signature(forward)); print((forward.__doc__ or '')[:1500])
except Exception as e: print(e)
# token lengths
lens=[len(tok(q['question'], p, truncation=True, max_length=512)['input_ids']) for q in practice for p in q['paragraphs']]
print("tok len mean/median/max", np.mean(lens), np.median(lens), max(lens))
# marginal costs of components (pure bookkeeping, no passes)
base=cost(); print(base)
remove_head(5,0); print('head', cost()); restore_head(5,0)
remove_neurons(5,[0]); print('neuron', cost()); restore_neurons(5,[0])
remove_layer(5); print('layer', cost()); restore_layer(5)
print(edited())
NOTES.write_text("# Notes\n\nSession start. Baseline cost: %s\n" % (base,))
```