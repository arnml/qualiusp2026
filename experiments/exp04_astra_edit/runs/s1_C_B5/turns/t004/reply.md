The weights have no obvious zeroed-out blocks. Later layers have smaller output paths, but that alone is insufficient evidence to remove them. The first pass will measure neuron activity and inspect attention on gold paragraphs with little title overlap and distractors with strong title overlap.

```python
# Hypothesis: weakly active MLP paths and attention focused on structural tokens are redundant for relevance ranking.
import re

def title_overlap(q, p):
    words = set(re.findall(r"\w+", practice[q]["question"].lower()))
    title = set(re.findall(r"\w+", practice[q]["paragraphs"][p].split(".")[0].lower()))
    return len(words & title) / max(1, len(title))

diagnostic_items = []
for q in selection_q[:4]:
    gold_p = [p for p, g in enumerate(practice[q]["gold"]) if g]
    other_p = [p for p, g in enumerate(practice[q]["gold"]) if not g]
    diagnostic_items.append((q, min(gold_p, key=lambda p: title_overlap(q, p))))
    diagnostic_items.append((q, max(other_p, key=lambda p: title_overlap(q, p))))
baseline_items = diagnostic_items + [x for x in selection_items if x not in set(diagnostic_items)]
baseline = forward(items=baseline_items, model="original", cls=True, mlp=True, attn_n=8)

def describe(value, depth=0):
    if hasattr(value, "shape"):
        return {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): describe(v, depth+1) for k, v in list(value.items())[:10]}
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "first": describe(value[0], depth+1) if value else None}
    return str(value)[:160]

print("Returned structure:", describe(baseline))
print("Attention examples:", [
    {"q": q, "p": p, "gold": int(practice[q]["gold"][p]),
     "type": practice[q]["type"], "title_overlap": round(title_overlap(q, p), 3)}
    for q, p in diagnostic_items
])
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nPass 1: original model on the selection half only; CLS, MLP activations, "
            "and attention for eight examples requested. Confirmation remains unexamined.\n")
```
