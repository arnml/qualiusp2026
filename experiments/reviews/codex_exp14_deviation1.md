# Codex adversarial review: exp14 deviation 1 (2026-11-16, before any ranking)

Command: `codex exec -s read-only` on PROTOCOL.md (deviation 1), make_notebook.py and the analyze.py diff.

> **Yes, conditionally.** The new ranking of `router_train` is a material change: it changes the comparison router's training data, so the secondary comparison is no longer "exp13's hop router refit exactly as in exp13." Re-ranking `dev_fresh` is descriptive and does not threaten the `b_eval` tests, but it is not an exact reuse of exp13's rankings.
>
> The notebook has a dangerous resume path: it skips training when model files already exist, but writes `model_sha.json` only when the current hashes differ from the registered hashes. A stale file may therefore describe different models, or be absent while skipped models are nonregistered, so `analyze.py` can apply the wrong hash assertions. Also, the piped training command lacks `pipefail`; a failed training process can be hidden by `tail` succeeding.
>
> **Before running:** preregister that the hop-router secondary is exploratory and trained on newly generated rankings, and make the notebook always record hashes for the actual models used, including on resume, then verify those hashes match every ranking input. Use a fresh output folder or reject stale artifacts.

**Response:** all three adopted (PROTOCOL.md, "After the Codex review of this deviation").
