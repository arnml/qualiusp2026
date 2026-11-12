# Hypothesis: redundant components have weak task-related CLS activity or attend mainly to structural tokens.
import re

def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

probe_items = []
bridge_q = [q for q in discovery_q if practice[q]["type"] == "bridge"]
for q in (bridge_q + discovery_q)[:4]:
    example = practice[q]
    overlap = [len(words(example["question"]) & words(p)) for p in example["paragraphs"]]
    gold = [p for p, g in enumerate(example["gold"]) if g]
    distractors = [p for p, g in enumerate(example["gold"]) if not g]
    probe_items.extend([(q, min(gold, key=lambda p: overlap[p])),
                        (q, max(distractors, key=lambda p: overlap[p]))])
probe_items = list(dict.fromkeys(probe_items))
discovery_items = probe_items + [x for x in discovery_items if x not in set(probe_items)]

baseline = forward(items=discovery_items, model="original", cls=True, mlp=True, attn_n=8)

def describe(value, depth=0):
    if hasattr(value, "shape"):
        return {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "first": describe(value[0], depth + 1) if value else None}
    return str(value)[:120]

print(describe(baseline))
print("Attention probe labels:", [(q, p, int(practice[q]["gold"][p])) for q, p in probe_items])
with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nWeight inspection: no exactly zero tensors or obvious deleted components. "
            "Pass 1 measures all discovery paragraphs, with CLS, MLP, and eight attention probes. "
            "Probes pair weak lexical-overlap gold paragraphs with strong-overlap distractors.\n")
