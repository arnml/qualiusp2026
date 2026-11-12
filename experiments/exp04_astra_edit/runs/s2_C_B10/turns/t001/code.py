# Hypothesis: weight magnitudes and exact redundancies can identify promising removals before using a forward pass.
import numpy as np
import torch

rng = np.random.default_rng(20261112)
split = rng.permutation(len(practice))
discovery_q = sorted(split[:120].tolist())
confirmation_q = sorted(split[120:].tolist())
discovery_items = [(q, p) for q in discovery_q for p in range(len(practice[q]["paragraphs"]))]
confirmation_items = [(q, p) for q in confirmation_q for p in range(len(practice[q]["paragraphs"]))]

print("Questions:", len(discovery_q), "discovery;", len(confirmation_q), "confirmation")
print("Paragraphs:", len(discovery_items), "discovery;", len(confirmation_items), "confirmation")
print("Initial edits:", edited())
print("Initial cost:", cost())
for name in NAMES:
    w = load_original(name)
    print(name, tuple(w.shape), "rms", round(float(w.square().mean().sqrt()), 6),
          "zero_fraction", round(float((w == 0).float().mean()), 4))

NOTES.write_text(
    "# Pruning experiment\n\n"
    "Goal: reduce encoder FLOPs while preserving both-gold@3 within one percentage point.\n"
    "Split: seeded permutation (20261112), 120 discovery questions and 80 confirmation questions. "
    "Confirmation questions will not guide removal selection.\n"
    "Methods permitted and planned: weight statistics, activation statistics, attention inspection, "
    "and paired causal removal experiments; no training, gradients, or fitted objectives.\n"
    "Forward budget: 10 calls. Inspect weights before choosing diagnostic probes.\n",
    encoding="utf-8",
)
