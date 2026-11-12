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
