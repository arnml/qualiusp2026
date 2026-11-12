I’ll split the practice questions into selection and confirmation halves, inspect the weights, then use paired removal experiments to test candidates. I’ll reserve the confirmation half for checking the chosen edit and keep the reasoning in `/work/notes.md`.

```python
# Hypothesis: weight structure can reveal redundant components before spending a forward pass.
import numpy as np
import torch
from pathlib import Path
from collections import Counter

NOTES = Path("/work/notes.md")
NOTES.write_text(
    "# Pruning notes\n"
    "Goal: reduce encoder FLOPs while preserving both-gold@3 within one percentage point.\n"
    "No training, gradients, fitted objectives, or independent model execution.\n"
    "Plan: deterministic 100-question selection / 100-question confirmation split; "
    "reserve confirmation outcomes until a candidate is chosen.\n"
)
rng = np.random.default_rng(20261112)
order = rng.permutation(len(practice))
selection_q = sorted(order[:len(order)//2].tolist())
confirmation_q = sorted(order[len(order)//2:].tolist())
all_items = [(q, p) for q, ex in enumerate(practice) for p in range(len(ex["paragraphs"]))]
selection_items = [(q, p) for q, p in all_items if q in set(selection_q)]
confirmation_items = [(q, p) for q, p in all_items if q in set(confirmation_q)]

print("Questions/items:", len(practice), len(all_items))
print("Selection/confirmation:", len(selection_q), len(confirmation_q))
print("Current edit:", edited())
print("Baseline cost:", cost())
print("Tensor count:", len(NAMES))
print("Representative tensors:")
for name in NAMES:
    if ".layer.0." in name or not any(f".layer.{i}." in name for i in range(6)):
        w = load_original(name)
        print(name, tuple(w.shape))
```


