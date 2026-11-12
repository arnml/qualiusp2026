# exp04: Astra edits the selector

Design and rationale: [DESIGN.md](DESIGN.md). The editor's prompt (Track C): [prompt_trackC.md](prompt_trackC.md).

## Files

| File | Role |
|---|---|
| `editlib.py` | Edit state (head / neuron / layer removal, row/column scaling, raw edits for Track R), scope checks, FLOPs/params cost, building the edited model. Removal is exact: verified against hook-based references (transformers 5.22 ignores `head_mask`). |
| `kernel.py`, `kernel_api.py` | The editor's persistent Python kernel and the functions it sees (`load`, `remove_*`, `scale_*`, `cost`, `edited`, `forward`, `practice`, `tok`). |
| `harness.py` | Turn loop. It also runs the budgeted forward-pass server (HTTP on localhost, a separate process from the kernel), drives the Codex editor (`codex exec -m gpt-6-astra`, all built-in tools and MCP servers off), applies the code audit, and logs everything. |
| `baselines.py` | Gradient-free and gradient pruning baselines at matched FLOPs (layer drop, Taylor, Wanda-style, random). |
| `dummy_script.json` | A scripted editor for dry runs. |
| `models/a1/` | A1 weights (git-ignored; sha256 prefix `605c986eb9a9a82b`). |
| `runs/<name>/` | One folder per run (git-ignored until results are final). |

## Run

```bash
uv sync
uv run python harness.py --run dry2 --track C --budget 2 --turn-cap 10 --editor dummy --script dummy_script.json
uv run python harness.py --run s1_C_B5 --track C --budget 5 --turn-cap 40 --editor astra
uv run python baselines.py --eval-n 300
```

## Per-run log (`runs/<name>/`)

- **Setup:** `prompt.md`, `config.json`, `practice.json`
- **Every turn,** in `turns/tNNN/`:
  - `msg_in.md`: what the editor received
  - `events.jsonl`: the Codex JSON events, used for the tool audit
  - `reply.md`, `code.py`, `output.txt`
- **Forward passes:** `forward_log.jsonl`, `fwd/pass_NNN.npz` (every pass)
- **Final state:** `work/edited/` (the submitted edit) and `work/notes.md` (the editor's notebook)
- **Records:** `run.json` (per-turn usage, timings, audit flags) and `summary.json`

## Isolation and audit

- **Codex:** the shell, unified exec, apps, browser, computer use, image generation, multi-agent, plugins, view_image, code mode, `node_repl` and `MCP_DOCKER` MCP servers, and web search are all disabled. Any Codex item other than `agent_message`, `reasoning` or `error` is logged as a violation. A probe confirmed that the model answers "NO_TOOLS" under this configuration.
- **Code audit:**
  - *Hard bans* (refused before execution): backward/autograd/optimizers, loading model classes, subprocess/socket/network, touching harness internals.
  - *Soft flags* (executed, recorded): `lstsq`, regression/classifier `.fit`, re-implementing layers.
- **Not isolated like the blog:** Windows has no bubblewrap, so the kernel runs in the same uv environment on the host. Rule compliance is enforced by the audit and the logs, not by an OS sandbox. This is reported as a limitation.

## Baselines (2026-11-12, `results/baselines_dev300.json`)

Setup: importance estimated on the editor's 200 practice questions; quality = dev both-gold@3 over 300 questions; A1 unedited = **0.970 at 39.26 GFLOPs** per question. The layer-drop rows' GFLOPs differ because whole layers are coarse.

| Encoder FLOPs kept | Layer drop | Taylor (grad) | **Wanda-style** | Random |
|---|---|---|---|---|
| ~90% (35.2 GF) | 0.767 *(32.8 GF)* | 0.943 | **0.970** | 0.953 |
| ~80% (31.2 GF) | 0.290 *(26.3 GF)* | 0.927 | **0.950** | 0.913 |
| ~70% (27.5 GF) | 0.290 *(26.3 GF)* | 0.877 | **0.930** | 0.887 |
| ~60% (23.6 GF) | 0.020 *(19.9 GF)* | 0.823 | **0.917** | 0.797 |
| ~50% (19.5 GF) | 0.063 *(13.4 GF)* | 0.720 | **0.820** | 0.443 |

**Readout**
- **Wanda-style is the strongest baseline at every level:** −0, −2, −4, −5 and −15 points.
- **Neurons are very redundant:** 4,586 of 9,216 MLP neurons plus 38 of 72 heads can go at a 15-point cost.
- **Whole layers are not removable:** −1 layer costs 20 points; −3 layers collapses the model.
- **Taylor** is no better than random until 50%, probably because gradients over 200 questions are noisy.
- **Bar for Astra:** at ~60% FLOPs, beat 0.917; at ~50%, beat 0.820.

## Dry run (2026-11-12, dummy editor, `runs/dry2`)

Everything worked:
- the kernel API answered and the practice data loaded
- a full practice pass took ~5 min on CPU (1,992 pairs, both models, [CLS] + MLP + attention)
- A1 practice both-gold@3 = 0.945
- the `.backward` call was refused before running
- removing layer 5 + 1 head + 3 neurons → 32.6 GFLOPs (from 39.3) and both-gold@3 0.745
- the budget was enforced, forward was refused once the budget was spent, and the session ended after the final reply

## Stage 1 result (`runs/s1_C_B5`, 2026-11-12): Astra, Track C, B = 5

**Run:**
- 11 turns, 5/5 passes, **0 tool violations, 0 refused code**
- 8 min of editor time; Codex tokens: 1.20M input / 80k output

**Astra's process, from its logs:**
1. **Setup and weights:** fixed a 100/100 practice split (selection / confirmation, seed 20261112), then read static weight statistics.
2. **Pass 1, diagnosis:** ran the original model on the selection half, collecting [CLS], MLP activations and 8 attention maps.
3. **Pass 2, Candidate A:** removed 3,584 MLP neurons, ranked by [CLS] activation RMS × output-weight norm. Per layer: 384 / 512 / 512 / 512 / 640 / 1,024. Its rationale for removing the most in the last layer: only its [CLS] reaches the scorer. Result on selection: 92 → 93/100.
4. **Pass 3, Candidate B = A + 8 "weak or CLS-only" heads:** 88/100, **rejected**. The attention pattern was a poor guide to head importance.
5. **Pass 4, Candidate C = 4,480 neurons:** 88/100, **rejected**.
6. **Pass 5, confirmation of A on the untouched half:** 97 → 96/100. On all 200 practice questions: 189 → 189.

**Held-out evaluation** (dev questions never seen by Astra; `evaluate.py`, matched baselines from `baselines.py --fractions 0.7616`):

| Selector | Encoder GFLOPs | dev300 both-gold@3 |
|---|---|---|
| A1 unedited | 39.26 | 0.970 |
| **Astra stage 1 (3,584 neurons)** | **29.90 (−23.8%)** | **0.963** |
| Wanda-style, same FLOPs | 29.78 | 0.950 |
| Taylor, same FLOPs | 29.77 | 0.927 |
| Random, same FLOPs | 29.90 | 0.913 |
| Layer drop (nearest) | 26.34 | 0.290 |

**Readout:** Astra meets the ≤1-point target on held-out questions (−0.7) and is above the best baseline at equal FLOPs (+1.3 points over Wanda). But 1.3 points is 4 questions out of 300. The 1000-question dev check is running; see below.

**Full dev (1000 questions):** Astra stage 1 = **0.946** both-gold@3. Unedited A1 and the matched baselines on the same 1000 questions are *running* (`baselines.py --eval-n 1000 --fractions 0.7616` → `results/baselines_dev1000.json`). The first 300 dev questions are easier than the full set, so compare only within the same set.

**Stage gate (pre-registered):** go to stage 2 if Astra ≥ the best baseline at equal FLOPs on dev. On dev300 it holds (0.963 vs 0.950). Final decision after the dev1000 comparison.

## Harness deviations log

- **2026-11-12, stage 1 (`s1_C_B5`), turn 1:** the prompt names `/work/notes.md` (copied from the Exp 8 Linux wording). On Windows, `Path("/work/...")` resolves to `C:\work`, so Astra's first cell failed with FileNotFoundError. The kernel's `NOTES`/`WORK` variables point to the right folder. Not interrupted mid-run. **Fix for stages 2–3:** the prompt will say to use the `NOTES` / `WORK` variables, and the kernel will map `/work` paths.
- **Same run, later turns:** Astra created `C:\work\` itself and wrote `notes.md` and `removal_plan.json` there. These are copied into `runs/s1_C_B5/work/from_C_work/`. The folder `C:\work\` was left in place: removing a root-level folder needs the PI's approval. **Fixed for stage 2+:** the prompt now says to use the `WORK` / `NOTES` variables only, and that absolute paths like `/work` don't exist.
