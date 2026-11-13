I’ll use a fixed split of 100 questions to diagnose and choose an edit, then 100 untouched questions to confirm it. I’ll reserve the last forward pass for confirmation and use the middle passes to test the proposed mechanism at different strengths.

```python
# Hypothesis: lexical-match signals cause distractors to outrank weakly linked gold paragraphs, and their source can be localized in the layer activations.
import numpy as np
import torch
from pathlib import Path

rng = np.random.default_rng(20261113)
order = rng.permutation(len(practice))
discovery_ids = np.sort(order[:len(order)//2])
confirmation_ids = np.sort(order[len(order)//2:])
all_items = [(qi, pi) for qi, q in enumerate(practice) for pi in range(len(q["paragraphs"]))]
discovery_items = [(qi, pi) for qi, pi in all_items if qi in set(discovery_ids.tolist())]
confirmation_items = [(qi, pi) for qi, pi in all_items if qi in set(confirmation_ids.tolist())]
item_row = {item: r for r, item in enumerate(all_items)}

NOTES.write_text(
    "# Weight-edit experiment\n"
    "Fixed random split, seed 20261113: 100 discovery questions and 100 confirmation questions "
    "(or equal halves if the supplied count differs). Confirmation outcomes will remain unexamined "
    "until the candidate and its strength are fixed.\n"
    "Plan: baseline with representations; three discovery interventions; final paired confirmation.\n"
    "No optimization, fitted predictive models, or private model execution. "
    "Only descriptive statistics, weight inspection, and the provided forward function.\n"
)
baseline = forward(items=all_items, model="original", cls=True, mlp=True, attn_n=8)
print("Questions/items:", len(practice), len(all_items))
print("Discovery/confirmation questions:", len(discovery_ids), len(confirmation_ids))
print("Returned fields:", {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in baseline.items()})
print("Editable tensor names:", [n for n in NAMES if n.startswith("bert.encoder.layer.5.") or n.startswith("bert.pooler.")])
```


