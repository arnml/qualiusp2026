My previous reply used tool calls instead of a fenced code block, so nothing ran and no passes were used. It also strung together steps whose outputs I hadn't seen. From here on I'll send one block per turn and decide each next step from the actual output.

```python
# Hypothesis: none yet — inventory the data, tensors and notes before spending any forward pass.
import numpy as np, collections, json, re, torch, pandas as pd
print("WORK:", WORK, "| NOTES exists:", NOTES.exists())
if NOTES.exists(): print(NOTES.read_text()[:1500])
print("n tensors:", len(NAMES)); print([n for n in NAMES if "layer.5" in n or "pooler" in n])
print("n questions:", len(practice), "types:", collections.Counter(q["type"] for q in practice))
npar = [len(q["paragraphs"]) for q in practice]; print("paragraph counts:", collections.Counter(npar), "total:", sum(npar))
for qi in [0, 1]:
    q = practice[qi]; print("\nQ:", q["question"], "| A:", q["answer"], "|", q["type"])
    for p, g in zip(q["paragraphs"], q["gold"]): print(" ", g, p[:110].replace("\n", " "))
lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for q in practice for p in q["paragraphs"]]
print("para token len: mean %.1f, median %d, max %d" % (np.mean(lens), np.median(lens), max(lens)))
print("edited():", edited())
```