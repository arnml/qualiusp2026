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

## Stage 3 result (`runs/s3_C_B50`, 2026-11-12): Astra, Track C, B = 50, lossless target

Pre-registered in `8a39213` (DESIGN.md, "Amendment before stage 3"). Prompt: `prompt_trackC_s3.md` (target within 0.5 points of A1).

**Run:** 23 turns, 50/50 passes, **0 tool violations, 0 refused code**, 5.85M Codex input / 210k output tokens, ~2 h wall clock (14 min of editor time; the rest is CPU forward passes). Codex weekly usage went from 19% to 22% (5-hour window: 16%).

**Final edit:**
- **Removed:** 2,016 MLP neurons, no heads or layers.
  - 96 low-norm neurons in each of layers 0–4
  - **the whole MLP of layer 5** (1,536 neurons)
- **Cost:** 34.00 GFLOPs (−13.4%), 9.10M encoder params (−14.6%).
- **Astra's evidence:** the layer-5 MLP is unnecessary for the ranking, but dropping the whole of layer 5 hurts badly, so layer 5's attention still matters.
- **Practice:** 94.5% → 95.0% (selection 111 → 112/120; confirmation 78 → 78/80, the same questions right).
- **Caveat Astra stated itself:** it also used the confirmation half to roll back larger cuts, so that half was not untouched at the end.

**Held-out, dev1000** (`runs/s3_C_B50/eval.json`, `results/baselines_dev1000_matched_s3.json`):

| Selector | GFLOPs | heads / neurons removed | dev1000 both-gold@3 |
|---|---|---|---|
| A1 unedited | 39.26 | — | 0.959 |
| **Astra stage 3** | **34.00** | 0 / 2,016 | **0.957** |
| Wanda-style, same FLOPs | 33.99 | 7 / 1,473 | 0.962 |
| Taylor, same FLOPs | 34.00 | 9 / 1,313 | 0.945 |
| Random, same FLOPs | 34.00 | 9 / 1,313 | 0.933 |

**Paired McNemar test, Astra vs Wanda** (`runs/s3_C_B50/mcnemar_dev1000.json`): Astra-only correct 9, Wanda-only correct 14, p = 0.40.

**Dense Wanda curve on dev1000** (`results/wanda_curve_dense_dev1000.json` plus the earlier files):

| FLOPs kept | 70% | 72.8% | 75% | 77.5% | 80% | 82.5% | 85% | 86.6% | 87.5% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| GFLOPs | 27.48 | 28.56 | 29.45 | 30.37 | 31.22 | 32.20 | 33.22 | 33.99 | 34.35 | 35.33 |
| both-gold@3 | 0.930 | 0.940 | 0.938 | 0.952 | 0.953 | 0.954 | 0.959 | 0.962 | 0.956 | 0.958 |

The curve is not monotone above 80%: the differences there are within noise (±0.7 points).

**Test400, params and CPU latency** (`runs/s3_C_B50/final_eval.json`, `final_eval.py`):
- Removed neurons and heads are sliced out of the matrices; scores match the zeroed model to ≤ 1.2e-5.
- Latency: torch with 8 CPU threads, one question's paragraphs per batch, mean over 200 dev questions, median of 3 repeats.

| Selector | GFLOPs | Encoder params | test400 both-gold@3 | CPU ms per question |
|---|---|---|---|---|
| A1 unedited | 39.26 | 10,646,784 | 0.940 | 241 |
| **Astra stage 3** | 34.00 | **9,096,480** | 0.9375 | 219 (−9.1%) |
| Wanda, same FLOPs | 33.99 | 9,169,311 | 0.9425 | 217 (−10.0%) |

**Pre-registered verdict:**
1. **Lossless check: passed.** dev1000 0.957 ≥ 0.954; −0.2 points vs A1 on dev, −0.25 on test.
2. **Primary, cost at equal quality: TIE (leaning Wanda).** Wanda first reaches 0.957 at ~32.8 GF (interpolating 82.5% → 85%), against Astra's 34.00 GF. That is 1.2 GF, under the pre-registered 2.0 GF margin.
3. **At equal FLOPs:** Wanda 0.962 vs Astra 0.957 on dev (p = 0.40) and 0.9425 vs 0.9375 on test; both differences within noise. Astra's model has 0.8% fewer params; latency is the same within 2 ms.

**exp04 Track C overall (stages 1–3):**
- **Cheaper than A1:** an LLM editing weights with only 5–50 forward passes reliably produces selectors that are 13–27% cheaper, near or at A1's quality. CPU time drops by ~9% at −13% FLOPs.
- **Not better than Wanda:** the gradient-free Wanda-style baseline matches it at every stage. More forward passes did not open a gap.
- **Qualitatively different solutions:**
  - Astra never removes heads.
  - It found that the last layer's MLP can go entirely while its attention must stay.
  - It audits its own cuts with paired tests on held-out practice questions.
- **For the thesis:** an honest negative-to-neutral result on "LLM weight editing beats classical pruning", with a positive engineering result (a cheaper selector at the same quality) and an interpretable mechanism finding.

## Track R, stage R1 (`runs/r1_R_B5`, 2026-11-13): Astra edits A0, B = 5

Pre-registered in `25bc080` (DESIGN.md, "Track R pre-registration"). The start model is A0, the off-the-shelf encoder; practice = A2's 200 training questions.

**Run:** 16 turns, 5/5 passes, **0 tool violations, 0 refused code**, 2.78M Codex input tokens, 9 min of editor time.

**Astra's process (from its notes):**
1. **Fixed split:** 100 discovery / 100 confirmation questions (seed 20261113). Confirmation was kept unexamined until the candidate was frozen.
2. **Mechanism candidate:** layer-4 MLP neurons that favour lexically overlapping distractors over weak gold paragraphs. Screened with within-question activation correlations, distractor-minus-gold means and output-weight alignment, all descriptive statistics.
3. **Intervention:** `W[:, S] = (1 − α) W_orig[:, S]` on 96 columns of `bert.encoder.layer.4.output.dense.weight` (36,864 entries), at α = 0.5 / 1.0 / 2.0 on discovery.
   - Tokens@90: 950 → 913 / 912 / 920.
   - bg3 unchanged.
   - Astra noted that the effect is not monotone in strength.
4. **Predeclared confirmation rule:** keep only if the token saving exceeds 2 SE and its 95% CI excludes 0. On confirmation, α = 1 gave 963 → 937 tokens (CI −18 to +49) and bg3 54% → 55%.
5. **The rule failed, so the edit was reverted. The submitted edit is empty (0 changed weights).**

**Pre-registered verdict: no repair.** The submitted model is A0, so dev1000 = A0 (0.593 bg3, 1,089 tok@90). The R1 gate ("dev1000 shows any change") is not met.

**Exploratory check, not pre-registered** (`runs/r1_R_B5_candidate_explore`): Astra's rejected α = 1 candidate, rebuilt from `work/confirmation_candidate.pt` and scored on dev1000.

| Model | dev1000 bg3 | dev1000 tok@90 |
|---|---|---|
| A0 | 0.593 | 1,089 |
| Rejected candidate | 0.594 | 1,097 (diff vs A0: +8, 95% CI −10 to +17) |
| A2 (same 200 questions, fine-tuned) | 0.769 | 698 |

McNemar vs A0: 4 vs 3, p = 1.0. **The candidate has no effect on held-out questions, so Astra's rejection was correct.** Its discovery-half gain (−38 tokens) was overfitting to 100 questions.

**Readout:**
- This matches the prior from the exp03 probe: A0 lacks the skill, and a 1% weight edit found by analysis does not add it.
- **The editor behaved like a careful experimenter:**
  - a predeclared split and rule
  - an honest null submission, rather than shipping an unconfirmed gain
