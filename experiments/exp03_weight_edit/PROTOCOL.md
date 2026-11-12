# exp03 protocol: LLM weight editing of a small cross-encoder selector

Status: **draft for review, not yet run.** Once approved, commit this file before any run; the commit hash acts as our pre-registration.

## 1. Question

Can a frontier LLM, working only by **reading and directly editing weights** (no gradient descent), improve the off-the-shelf 22.7M cross-encoder as a HotpotQA evidence selector? And does it get **closer to full fine-tuning than gradient fine-tuning on the same small data budget**?

| Arm | Selector | Data the method sees | Status |
|---|---|---|---|
| A0 | `ms-marco-MiniLM-L6-v2`, unchanged | none | done (exp02) |
| A2 | A0 + gradient fine-tune | N = 50 / 200 train questions | done (exp02) |
| **A3** | **A0 + LLM weight edit** | **same N questions, forward-pass budget B** | this experiment |
| A3-ctrl | A0 + random edit with the same entries and norm as A3 | — | this experiment |
| A1 | A0 + gradient fine-tune on 90k questions | ceiling | done (exp02) |

A4 (the editor also sees a large reranker's activations) is deferred until A3 has been seen.

## 2. Sources this protocol copies

**[Takehi et al. 2026](https://rikiyat.github.io/blog/weight-space-programming/)**, with harness details and the full Exp 8 prompt in the logs repo [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Their Exp 8 (the Qwen3.5-0.8B loop repair) is the closest template:
- **Harness:** Astra = `gpt-6-astra` through `codex exec` / `codex exec resume`, one call per turn, prompt on stdin. All built-in Codex tools are off (shell, file edit, web, sub-agents, plugins).
- **The model's only instrument** is a persistent, network-less Python kernel (bubblewrap, CPU only), with these functions: `NAMES`, `load(name)`, `load_original`, `save_edit`, `drop_edit`, `edited`, `tok`.
- **Kernel limits:** 900 s and 96 GB per execution, output cut at 10k characters, at most 300 code runs. Libraries: numpy, scipy, scikit-learn, torch-CPU, pandas, sympy, networkx, safetensors, tokenizers.
- **Forward passes** go through a separate tool with a total budget (10 / 25 / 50, one run per budget). One pass = one request of any size up to a cap; it can return activations; failed requests don't count.
- **Prompt sections, in order:** TASK · HOW IT WILL BE MEASURED · WHAT COUNTS · WHAT YOU HAVE · HOW TO WORK · RULES · HOW THE SESSION RUNS.
- **RULES:** no gradient descent or training (no backward passes, autograd, optimizers or iterative fitting to a loss); no running the model outside the tool; no simulating the network in the kernel; no internet.
- **WHAT COUNTS:** edits must come from understanding a mechanism. `save_edit` refuses edits to the final norm, the embeddings or the output layer, and any edit that only multiplies a whole tensor by a constant.
- **HOW TO WORK:** diagnose first; localize causally with a dose-response; use enough statistical power (SE ≈ 50/√n points); compare original and edited on the same items; keep an edit only if its gain beats its SE on items not used to choose it; state a hypothesis in each code block; keep `/work/notes.md`.
- **Session:** runs until the budget is spent; replying without code doesn't end it; handoff to a fresh conversation every 40 turns, carrying the task, the notebook and the current edit; whatever is in `/work/edited` at the end is submitted with an explanation.
- **Records:** every turn audited as tool-free, token usage per turn in `run.json`, exact edit diffs released with `apply_edit.py`, base revision pinned by hash.
- **Their findings that shape our expectations:** the successful edit removed a failure mode (a "loop direction" in the MLPs of layers 15 and 19; 64 columns × 2 layers = 131,072 weights). On Gemma, which had no such failure mode, no edit helped. The 10- and 25-pass runs barely worked. They ran one run per budget, without variance.

**[Ahmed 2026, arXiv:2611.08070](https://arxiv.org/abs/2611.08070)**, an audit of Claude Opus 5.5 reading CNN weights:
- fresh, isolated sessions (Claude Code non-interactive, xhigh effort, tools/MCP/memory off, empty folder), with identical wording across conditions
- scoring rules written before sending prompts; quoted numbers checked against the real parameters; partial-tracing claims re-checked over the whole network
- controls: label permutation, shuffled weights, familiarity (unusual architectures and datasets), training amount
- its own proposed next tests: a forward-pass test, and blind weight editing without gradient descent (our experiment)
- weaknesses to avoid: single seed, no second rater, no confidence intervals, confounds introduced by the prompt itself

## 3. Network (what the editor is told)

`cross-encoder/ms-marco-MiniLM-L6-v2`: BERT, 6 layers, width 384, 12 heads of size 32, GELU MLP of width 1536, vocab 30,522, max 512 positions.

```
input  = [CLS] question [SEP] paragraph [SEP]   (token_type 0 for question, 1 for paragraph)
h      = BERT encoder (6 layers)
score  = classifier( tanh( pooler.dense( h[CLS] ) ) )    -> one logit; higher = more relevant
```

- Tensor names follow `bert.encoder.layer.{0..5}.{attention.self.{query,key,value}, attention.output.dense, intermediate.dense, output.dense, *.LayerNorm}.{weight,bias}`, plus `bert.pooler.dense` and `classifier`.
- **Editable:** encoder layers and pooler, about 10.8M weights.
- **Out of scope** (`save_edit` refuses): embeddings (word, position, token-type and their LayerNorm), `classifier.*`, and any edit that is a scalar multiple of a whole tensor.
- **Edit-size cap:** at most **1% of editable weights (≈108k entries)** may differ from the original. That's the same order as the Qwen edit (131k).

Why the classifier is excluded: otherwise the trivial solution is to fit a linear probe on [CLS] and write it into `classifier.weight`, which is just training a logistic regression. We want internal mechanisms.

## 4. Task data and measurement

- **Practice data (the only data the editor sees):** the same N HotpotQA *train* questions as A2 (`load_train(N, seed=0)`; N = 50 and N = 200), with their 10 paragraphs and gold labels.
- **Never seen by the editor:** the 1000 dev questions (threshold selection) and the 400 test questions (report).
- **Primary metric:** test both-gold at the dev-chosen threshold for 0.90 dev both-gold, reported as tokens. This is the same rule as exp02's `analyze.py`, so A3 drops straight into the exp02 table. Lower is better.
- **Secondary metrics:** test top-2 both-gold; the dev-threshold curve; F1 with gpt-5.4-mini on the 400 questions (`generate.py`) for the best A3 run, A2 at the same N, and the random-edit control.
- **Success, written before running:**
  - A3 beats A2 at the same N on the primary metric, with a paired bootstrap 95% CI over test questions that excludes 0;
  - the random-edit control does not;
  - and switching the edit off (restoring the edited entries) loses the gain.

## 5. Budgets and rules

| Item | Value | Note |
|---|---|---|
| Forward-pass budget B | 10, 25, 50 | as in Exp 8 |
| One forward pass | up to 2,048 (question, paragraph) pairs from the practice set, or up to 64 pairs the editor writes itself | optionally returns per-layer [CLS] hidden states, MLP activations, attention for ≤ 8 pairs |
| Kernel | persistent Python, CPU, no network; ≤ 300 code runs; ≤ 900 s per run; output ≤ 10k chars | |
| Turns | until the budget is spent, or 200 turns; handoff every 40 turns | |
| Forbidden | backprop/autograd/optimizers; iterative fitting to a loss; **closed-form solving of a regression/classification objective on the practice data** (lstsq, logistic regression, SVM) whose result is written into weights; running the network outside the tool; simulating the network in the kernel; internet | the third item is stricter than Exp 8 (see §7) |
| Allowed | any analysis of weights, activations and pass results (means, differences, PCA, SVD, correlations, hypothesis tests); any direct, computed, in-scope edit | |

## 6. Runs

- **Editors:**
  - **Opus 5.5** through Claude Code non-interactive (`claude -p`), all built-in tools/MCP/memory off, xhigh effort. The only tool is our kernel bridge.
  - **Astra** through `codex exec` (if the account has access), built-in tools off.
- **Grid:** 2 editors × N ∈ {50, 200} × B ∈ {10, 50} × **3 seeds** (different practice-set order and a fresh session). That's 24 runs. Start with Opus × N = 200 × B = 50 × 1 seed, as a pilot.
- **Identical wording** across runs; only N, B and the budget lines change.
- **Logged per run:** every prompt and reply, every code block with its output, every forward-pass request and response, token usage, wall-clock time, and the final edit as an exact diff (`edit.safetensors` + `apply_edit.py` with a base-model hash check). An audit script confirms that no turn used a tool other than the kernel.

## 7. Known threats and how we handle them

| Threat | Handling |
|---|---|
| "Editing" turns into training (e.g. writing a fitted probe into weights) | classifier excluded; closed-form fitting to a loss forbidden; every code block logged; the audit flags `lstsq`/`LogisticRegression`/`.backward` and the like. Mean-difference directions are allowed and reported as such. |
| The gain is just regularisation noise | random-edit control on the same entries with the same Frobenius norm |
| The editor's explanation is post-hoc | turn the edit off, and apply it only in the layer the editor claims matters |
| Leakage of the eval set | the kernel holds only the N practice questions; dev and test live outside the sandbox |
| One lucky run | 3 seeds per cell; report mean and min/max |
| Familiarity (MS MARCO MiniLM is a famous model) | reported as a limitation; optional later check on a less known small cross-encoder |
| Cost | token usage per turn recorded; pilot first |

## 8. Prompt

See `prompt_template.md`. Placeholders: `{N}`, `{B}`, `{TURN_CAP}`.

## 9. Go/no-go before any agent run (~1 h, no LLM)

On the N = 200 practice questions, extract the [CLS] hidden state per layer for all pairs. Fit a linear probe offline (this is our check, not the editor's method) and measure, per layer, how well one direction separates gold from distractor (AUC). Also check whether A0's errors align with a lexical-overlap direction.
- **High AUC in middle layers** (the information is present but not used) → an edit is plausible → run the pilot.
- **Low AUC everywhere** → the obstacle is not a single direction → expect A3 ≈ A0; report that, and run only a minimal pilot.
