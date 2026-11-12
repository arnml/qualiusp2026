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

**Full dev (1000 questions; `results/baselines_dev1000_matched_s1.json`):**

| Selector | GFLOPs | heads / neurons removed | dev1000 both-gold@3 | vs A1 |
|---|---|---|---|---|
| A1 unedited | 39.26 | — | 0.959 | — |
| **Astra stage 1** | 29.90 | 0 / 3,584 | **0.946** | −1.3 |
| Wanda-style | 29.78 | 19 / 2,147 | 0.945 | −1.4 |
| Taylor | 29.77 | 23 / 1,839 | 0.923 | −3.6 |
| Random | 29.90 | 16 / 2,334 | 0.895 | −6.4 |
| Layer drop | 26.34 | 2 layers | 0.291 | −66.8 |

**Readout (honest):**
- On 1000 questions, **Astra ties with Wanda** (0.946 vs 0.945, one question apart). The dev300 advantage (+1.3) does not hold up at the larger sample.
- Both lose ~1.3 points, slightly more than the 1-point target.
- **The two reach the same quality with different solutions:** Astra removed neurons only and kept every head; Wanda removed 19 heads plus fewer neurons. That supports Astra's own finding that heads matter more than their attention patterns suggest.
- Both are clearly better than Taylor and random.
- **Paired McNemar test (Astra vs Wanda, per question, dev1000):** Astra-only correct 25, Wanda-only correct 24, exact p = 1.0. No difference; they disagree on ~5% of questions in both directions.

**Stage gate (pre-registered):** go to stage 2 if Astra ≥ the best baseline at equal FLOPs on dev.
- Formally met (0.946 ≥ 0.945), but it's a **tie**, not an advantage.
- Stage 2 (B = 10) tests whether more passes let Astra go *beyond* Wanda, for example by compressing further at the same quality.

## Harness deviations log

- **2026-11-12, stage 1 (`s1_C_B5`), turn 1:** the prompt names `/work/notes.md` (copied from the Exp 8 Linux wording). On Windows, `Path("/work/...")` resolves to `C:\work`, so Astra's first cell failed with FileNotFoundError. The kernel's `NOTES`/`WORK` variables point to the right folder. Not interrupted mid-run. **Fix for stages 2–3:** the prompt will say to use the `NOTES` / `WORK` variables, and the kernel will map `/work` paths.
- **Same run, later turns:** Astra created `C:\work\` itself and wrote `notes.md` and `removal_plan.json` there. These are copied into `runs/s1_C_B5/work/from_C_work/`. The folder `C:\work\` was left in place: removing a root-level folder needs the PI's approval. **Fixed for stage 2+:** the prompt now says to use the `WORK` / `NOTES` variables only, and that absolute paths like `/work` don't exist.

## Stage 2 result (`runs/s2_C_B10`, 2026-11-12): Astra, Track C, B = 10

**Run:**
- 15 turns, 10/10 passes, **0 tool violations**, 2.5M Codex input tokens.
- **Candidates Astra tested** (selection-half score /120; A1 = 112):
  - drop layer 1: 99
  - mlp_A, 3,456 neurons: 112
  - mlp_A + 8 heads: 102
  - mlp_B, 4,864 neurons: 109
  - mlp_B rescaled: 109
  - mlp_C, 4,096 neurons: 112
  - mlp_D, 4,608 neurons: 110
  - mlp_C + 2 heads: 110
- **Frozen edit: mlp_C.** 4,096 neurons removed (512/512/512/640/640/1,280 in layers 0–5), no heads or layers. **28.57 GFLOPs (−27.2%), 7.50M encoder params (−29.6%).**
- **Confirmation on the untouched half:** 77/80 vs A1 78/80. Astra itself reported that this narrowly misses the 1-point target.

**Held-out, dev1000** (`runs/s2_C_B10/eval.json`, `results/baselines_dev1000_matched_s2.json`):

| Selector | GFLOPs | Encoder params | heads / neurons removed | dev1000 both-gold@3 |
|---|---|---|---|---|
| A1 unedited | 39.26 | 10.65M | — | 0.959 |
| **Astra stage 2** | **28.57** | **7.50M** | 0 / 4,096 | **0.940** |
| Wanda-style, same FLOPs | 28.56 | 7.74M | 22 / 2,377 | 0.940 |
| Taylor, same FLOPs | 28.57 | — | 26 / 2,064 | 0.897 |
| Random, same FLOPs | 28.57 | — | 20 / 2,533 | 0.880 |

**Paired McNemar test, Astra vs Wanda** (`runs/s2_C_B10/mcnemar_dev1000.json`): Astra-only correct 31, Wanda-only correct 31, p = 1.0.

**Wanda cost curve on dev1000** (`results/wanda_curve_dev1000.json`), for the pre-registered primary criterion:

| FLOPs kept | 60% | 65% | 70% | 72.8% | 80% | 85% | 100% (A1) |
|---|---|---|---|---|---|---|---|
| GFLOPs | 23.55 | 25.52 | 27.48 | 28.56 | 31.22 | 33.22 | 39.26 |
| both-gold@3 | 0.909 | 0.916 | 0.930 | 0.940 | 0.953 | **0.959** | 0.959 |

**Primary criterion (cost at equal quality), honest readout:**
- **FLOPs:**
  - *Stage 2:* Wanda needs 28.56 GF to reach Astra's 0.940; Astra used 28.57 GF.
  - *Stage 1:* interpolating the curve, Wanda needs ~29.8 GF to reach 0.946; Astra used 29.90 GF.
  - **Not cheaper than Wanda; a tie at both stages.**
- **Parameters:** at equal FLOPs and quality, Astra's models are smaller:
  - stage 2: 7.50M vs Wanda 7.74M (−3.1%)
  - stage 1: 7.89M vs 8.06M (−2.1%)

  This holds because Astra removes neurons only (cheap in FLOPs, heavy in parameters), while Wanda also removes heads. It's a small edge, and a consequence of the unit choice rather than a better importance estimate.
- **Distance to A1:** Astra's stage 2 edit is −1.9 points, beyond the 1-point target. Its stage 1 edit was −1.3.
- **Lossless point:** Wanda keeps A1's full quality (0.959) at 85% FLOPs (−15%, 9 heads + 1,609 neurons). Neither Astra stage aimed there.
- **Noise:** ±0.7 points SE on 1,000 questions.

**Stage gate to stage 3** (pre-registered: Astra ≥ the best baseline at equal FLOPs): formally met, but again only as a tie, not an advantage. The PI decides whether to run B = 50.
