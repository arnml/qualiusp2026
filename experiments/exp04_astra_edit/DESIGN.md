# exp04 design: Astra edits the selector (compress, then repair), in stages

Status: **design for review, not run.** Supersedes running exp03 as pre-registered (see "Deviation" below).

## Why exp04 exists (deviation from exp03's no-go)

The exp03 probe (`../exp03_weight_edit/PROBE_EXPLAINED.md`) said no-go: A0 has no *linear* [CLS] direction that an edit could unlock.

The PI (Arnold) decided to run the Astra experiment anyway, for two reasons we record honestly:
1. **The probe is limited.** It only tests straight-line readouts at one position. Edits by a frontier model may work through attention, token-level or non-linear mechanisms that a linear probe cannot see. The Takehi et al. results were not intuitive either.
2. **The thesis goal is "smaller / cheaper".** So the edit should not only *activate* abilities but also *remove* redundant parts. Removal is exactly what worked in the blog: the 25-pass Qwen run zeroed an attention head, and the 50-pass edit subtracted a direction.

This is therefore **exploratory**, and the exp03 probe result is reported as the prior expectation.

## Objectives, checked against the thesis

| Thesis goal | What exp04 measures | Honest note |
|---|---|---|
| Fewer tokens to the LLM at equal quality | Track R: tokens at 90% both-gold (as exp02) | A1 already reaches 391 tok (oracle 323), so headroom is small for A1 and large for A0/A2 |
| Cheaper selector (FLOPs, params, CPU latency) | Track C: selector FLOPs / params / CPU ms per question at ≤1 pt both-gold loss | Selector FLOPs are ~1/180 of the generator FLOPs it saves; compression matters for **latency and CPU deployment** (A1 on CPU ≈ 0.4 s per question for 10 paragraphs), not for $ |
| A novel, defensible method | LLM weight editing vs gradient-free pruning baselines at the same budget | Novelty: LLM-driven editing applied to a retrieval selector |

## Two tracks

### Track C: compress (main; runs first)

- **Start model:** A1, the full fine-tune from exp02 (weights in Google Drive `MyDrive/qualiusp2026/exp02_results/full/model`; must be copied locally).
- **Astra's task:** remove as much computation as possible (attention heads, MLP neurons, whole layers; "remove" = zero the weights so that the component can be physically deleted) while keeping practice both-gold within 1 point of A1. Reported as structured sparsity, then **physically pruned** so that FLOPs and latency are real.
- **Baselines at the same removed fraction**, computed by us without gradients:

| Baseline | Source | Uses forward passes? |
|---|---|---|
| Drop top-k layers | Sajjad et al. 2023 \cite{sajjad2023droppinglayers}; Gromov et al. 2024 \cite{gromov2024deeperlayers} | no |
| Head importance by ablation (remove the least important heads) | Michel et al. 2019 \cite{michel2019sixteenheads}; Voita et al. 2019 \cite{voita2019heads} | yes, same budget B as Astra |
| Wanda-style score \|W\|·‖x‖ for neurons/heads, no retraining | Sun et al. 2024 \cite{sun2024wanda} | activation statistics from the same budget |
| Random removal (control) | — | no |

- **Success:** at equal FLOPs, Astra's model has higher dev both-gold than the best baseline. Or: at equal both-gold, it has fewer FLOPs.

### Track R: repair (secondary)

- **Start model:** A0 (off the shelf) or A2-n200. Same task as `../exp03_weight_edit/prompt_template.md`: fewer tokens at 90% both-gold.
- **Baseline:** A2 at the same N.
- Run only after Track C stage 1 shows the harness works.

## Staged schedule (5 → 10 → 50)

Each stage is one Astra run with a bigger forward-pass budget **B**, following the blog's budget ladder (they used 10/25/50; we use 5/10/50).

| Stage | B (forward passes) | Turn cap | Gate to the next stage |
|---|---|---|---|
| 1 | 5 | 40 | Harness works end to end; the edit is valid (in scope, applies, model runs). Any measurable change on dev. |
| 2 | 10 | 80 | Track C: Astra ≥ the best baseline at equal FLOPs on dev, or a clear trend from stage 1. Otherwise stop and report. |
| 3 | 50 | 200 | Full evaluation: dev curve, 400 test, CPU latency, then gpt-5.4-mini F1 on the 400 questions for the winner and its baseline. |

- **Offline evaluation costs nothing** (CPU scoring, ~10 min for dev + test), so every stage is evaluated on the full 1000 dev questions. Test (400) and LLM F1 only at stage 3, to keep the test set clean.
- **Practice data for Astra:** N = 200 train questions, the same as A2 and the exp03 probe.
- *Alternative reading of "5 → 10 → 50":* the number of practice questions. Not chosen, because 5 or 10 questions are too few to tell an edit from noise (SE ≈ 50/√n points, i.e. ±22 points at n = 5).

## Harness (copied from Takehi et al. Exp 8, released in RikiyaT/lm_understand-logs)

- **Editor:** `codex exec -m gpt-6-astra`, reasoning effort `xhigh` (Exp 8 used `max`), one call per turn, `codex exec resume <session>` for the next turn. Available on the PI's Codex account (`gpt-6-astra` is in the model list). Optional second editor: Opus 5.5 through `claude -p`.
- **All built-in Codex tools off** (shell, apply_patch, web, browser/computer use, apps, image generation, MCP, sub-agents). Each turn's JSON events (`--json`) are audited to confirm that no tool other than our kernel was used.
- **Turn loop:** Astra's reply contains one ```python block → our harness runs it in a **persistent sandboxed kernel** (CPU, no network) → the output (cut at 10k chars) becomes the next turn's input.
- **Kernel API:** `NAMES`, `load`, `load_original`, `save_edit`, `drop_edit`, `edited()`, `tok`, `practice`, `forward(...)` (budgeted; see the exp03 protocol §5), plus for Track C `remove(kind, layer, index)` helpers that zero a head, neuron or layer, and `cost()` reporting current FLOPs and params.
- **Rules:** as in exp03 (no gradients, no fitting written into weights, no simulating the network, no internet).
  - Track C additionally forbids changing **non-removed** weights by more than a rescale of the same row or column, so the track stays "removal".
  - Track R keeps the 1% edit-size cap.
- **Logging:** every prompt, reply, code block, kernel output and forward request; token usage; wall-clock time; the final edit as an exact diff plus `apply_edit.py` with a hash check.
- **Handoff:** every 40 turns, a fresh Codex session receives the task, `/work/notes.md` and the current edit.

## Prompt

The seven-section structure of `../exp03_weight_edit/prompt_template.md`. For Track C, only TASK, HOW IT WILL BE MEASURED and WHAT COUNTS change:
- *Goal:* "make the network cheaper by removing components (attention heads, MLP neurons, whole layers) that the task does not need."
- *Measure:* "FLOPs per question at the condition that held-out both-gold stays within 1 point of the unedited model; secondarily, both-gold at a fixed FLOPs budget."

The prompt will be written in full and committed before stage 1.

## What is needed before stage 1

1. ✅ **A1 weights locally** (2026-11-12): `models/a1/model.safetensors` (git-ignored), sha256 prefix `605c986eb9a9a82b`. Copied from Google Drive via Colab. The config and tokenizer were rebuilt from the base model. Local CPU scores reproduce the exp02 Colab scores (500 test pairs: max |diff| 0.0057, corr 1.000000, identical top-2 on 50/50 questions).
2. **Harness code:** about 300 lines (turn loop, kernel, forward server, audit), with a dry run using a dummy editor.
3. **Baselines** (CPU, no LLM): layer drop, head ablation, Wanda-style, random, all at the same removed fractions.
4. **Codex usage:** ChatGPT-plan limits, not dollars. The blog's 50-pass Qwen run took 20 h; ours is a 22M model, but stage 3 may still take hours.

## References

Takehi et al. 2026 \cite{takehi2026weightspace}; Ahmed 2026 \cite{ahmed2026weightspaceaudit}; Michel et al. 2019; Voita et al. 2019; Sun et al. 2024 (Wanda); Sajjad et al. 2023; Gromov et al. 2024; Arditi et al. 2024 (direction editing) \cite{arditi2024refusal}; probing: Alain & Bengio 2016; Hewitt & Liang 2019; Belinkov 2022; Marks & Tegmark 2023.

## Amendment before stage 2 (2026-11-12, recorded before the stage 2 run)

PI framing: **"similar score with fewer neurons or FLOPs is good; it is the same but cheaper."** Track C's success is therefore read primarily as **cost at equal quality**:

- **Primary:** for the editor's final edit, find the FLOPs each baseline (Wanda, Taylor, random) needs to reach the *same* dev1000 both-gold@3. That uses the baseline curve over fractions, interpolated. The editor wins if it needs fewer FLOPs, or fewer parameters, at the same quality.
- **Secondary:** quality at equal FLOPs (the stage 1 comparison) and the distance to the unedited model (target ≤ 1 point).
- **Reported for every selector:** encoder GFLOPs, encoder parameters, removed heads/neurons/layers, and later CPU latency after physical pruning.
- **Stage 1 under this framing:** Astra and Wanda reach the same quality (0.946 vs 0.945, McNemar p = 1.0) at ~23.8% fewer FLOPs than A1. A good result in itself (same selector, cheaper), but not cheaper than Wanda.
- **Stage 2:** B = 10 passes, turn cap 80, same prompt except the `/work` → `WORK`/`NOTES` fix (see README deviation log).
