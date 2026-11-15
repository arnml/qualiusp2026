Adversarial pre-run review. Read-only; do not modify files or run anything long. Read experiments/exp16_rank2_replication/PROTOCOL.md, data.py, train.py, evaluate.py, analyze.py, make_notebook.py, split_ids.json (and experiments/exp12_matched_control/README.md for the design being replicated).
Your role: argue for the NULL. Assume the experiment is designed (consciously or not) to confirm its authors' hopes. Find:
1. Any way the control is not actually matched (data, order, steps, schedule, truncation, tokenization lengths, hardware), or any leak/bug in the code that would bias the primary or secondary.
2. Statistical problems in the decision rules (power, gates, recovery-ratio CI, multiplicity, r* selection).
3. Alternative explanations that a positive result would not rule out, and what extra cheap check would rule them out.
4. Concrete changes to make BEFORE running (ranked, with exact edits where possible).
Max ~80 lines, condensed, specific file:line references.
