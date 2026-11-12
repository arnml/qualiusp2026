# Hypothesis: weight structure can reveal redundant components before spending a forward pass.
NOTES = Path("/work/notes.md")
NOTES.parent.mkdir(parents=True, exist_ok=True)
NOTES.write_text(
    "# Pruning notes\n"
    "Goal: reduce encoder FLOPs while preserving both-gold@3 within one percentage point.\n"
    "No training, gradients, fitted objectives, or independent model execution.\n"
    "Use a deterministic 100-question selection / 100-question confirmation split; "
    "reserve confirmation outcomes until a candidate is chosen.\n",
    encoding="utf-8",
)
rng = np.random.default_rng(20261112)
order = rng.permutation(len(practice))
selection_q = sorted(order[:len(order)//2].tolist())
confirmation_q = sorted(order[len(order)//2:].tolist())
selection_set = set(selection_q)
confirmation_set = set(confirmation_q)
all_items = [(q, p) for q, ex in enumerate(practice) for p in range(len(ex["paragraphs"]))]
selection_items = [(q, p) for q, p in all_items if q in selection_set]
confirmation_items = [(q, p) for q, p in all_items if q in confirmation_set]

print("Questions/items:", len(practice), len(all_items))
print("Selection/confirmation:", len(selection_q), len(confirmation_q))
print("Current edit:", edited())
print("Baseline cost:", cost())
print("Tensor count:", len(NAMES))
print("Representative tensors:")
for name in NAMES:
    if ".layer.0." in name or not any(f".layer.{i}." in name for i in range(6)):
        print(name, tuple(load_original(name).shape))
