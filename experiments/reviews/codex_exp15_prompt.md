Adversarial pre-run review. Read-only; do not modify files or run anything long. Read experiments/exp15_answer_f1/PROTOCOL.md, generate_f1.py, data.py, and experiments/exp02_cross_encoder/generate.py (the reader, prompt and F1 it loads), plus experiments/exp13_cascade/README.md for context.
Your role: argue for the NULL. Assume the experiment is designed to confirm its authors' hopes. Find:
1. Confounds in the F1 comparison (context length/tokens, paragraph order, truncation, reader sampling nondeterminism, prompt, answer normalization), and bugs in the code (index decoding, condition mapping, resumability, missing answers).
2. Statistical problems in the decision rules (power, subgroup choice, CI method, multiplicity).
3. Alternative explanations a positive result would not rule out, and cheap checks.
4. Concrete changes to make BEFORE running (ranked, exact edits where possible).
Max ~60 lines, condensed, file:line references.
