# Hypothesis: any token-saving benefit must reproduce on untouched questions and exceed paired sampling uncertainty.
crows = np.array([item_row[x] for x in confirmation_items])
cq = np.array([q for q, p in confirmation_items])
groups_c = [np.flatnonzero(cq == q) for q in confirmation_ids]
cgold = np.array([practice[q]["gold"][p] for q, p in confirmation_items], dtype=bool)
clength = np.array([len(tok.encode(practice[q]["paragraphs"][p], add_special_tokens=False))
                    for q, p in confirmation_items])

acceptance_rule = (
    "Freeze the strength with lowest discovery tokens@90 (ties favor smaller strength), "
    "then evaluate it once on confirmation questions. Keep only if confirmation token saving "
    "exceeds twice its paired bootstrap standard error, its 95% bootstrap interval excludes zero, "
    "and both-gold@3 does not show a deterioration exceeding twice its paired standard error. "
    "Otherwise restore the original tensor."
)
print(acceptance_rule)
print("Confirmation metadata prepared:", len(groups_c), "questions;", len(confirmation_items), "paragraphs.")
print("Token counts use the supplied tokenizer and exclude prompt wrappers; they are a retention-cost proxy.")
with NOTES.open("a") as f:
    f.write("\nPredeclared confirmation rule: " + acceptance_rule + "\n")
    f.write("Token-cost estimates use supplied tokenizer paragraph lengths, without prompt wrappers.\n")
