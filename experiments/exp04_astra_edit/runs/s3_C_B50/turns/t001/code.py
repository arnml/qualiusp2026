# Hypothesis: weight structure and paired activation diagnostics will reveal removable components without tuning on the confirmation split.
import numpy as np
import torch

rng = np.random.default_rng(831)
order = rng.permutation(len(practice))
select_q = sorted(order[:120].tolist())
confirm_q = sorted(order[120:].tolist())
select_items = [(q, p) for q in select_q for p in range(len(practice[q]["paragraphs"]))]
confirm_items = [(q, p) for q in confirm_q for p in range(len(practice[q]["paragraphs"]))]

print("Split:", len(select_q), "selection questions;", len(confirm_q), "confirmation questions")
print("Paragraph counts:", len(select_items), len(confirm_items))
print("Initial edits:", edited())
print("Initial cost:", cost())
print("Tensor names and shapes:")
for name in NAMES:
    if ".layer.0." in name or "pooler" in name or "classifier" in name:
        print(name, tuple(load_original(name).shape))

NOTES.write_text(
    "# Pruning experiment\n"
    "Goal: reduce encoder FLOPs with no measurable both-gold@3 loss.\n"
    "Split: numpy default_rng(831), permute all 200 questions; first 120 for selection, "
    "remaining 80 reserved for confirmation. No confirmation outcomes used to rank components.\n"
    "Allowed methods: weight inspection, activation statistics, and causal removal tests; no training.\n"
)
