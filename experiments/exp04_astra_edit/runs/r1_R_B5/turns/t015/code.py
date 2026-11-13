# Hypothesis: the observed confirmation gain is insufficient to justify changing the submitted model.
if torch.count_nonzero(load(target_name) != load_original(target_name)).item():
    drop_edit(target_name)
final_status = edited()
print("Final edited():", final_status)
print("Changed entries in tested tensor:",
      int(torch.count_nonzero(load(target_name) != load_original(target_name)).item()))
with NOTES.open("a") as f:
    f.write("\nFinal verification: original weights restored; no candidate retained. "
            "Confirmation saving was 25.96 tokenizer tokens/question, SE 15.90, "
            "95% paired bootstrap interval [-18.31, 49.49]. "
            "Both-gold@3 changed from 54% to 55% (one win, zero losses).\n")
