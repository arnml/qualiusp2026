# Session handoff (for context compaction): read this first

Last updated: 2026-11-12. Repo: `C:\code\qualiusp2026` (GitHub `arnml/qualiusp2026`, public, branch `main`).
User: Arnold Moya (master's qualification, USP; PhD path later). Previous paper: BRACIS 2026, Noise-Gate (code: `C:\code\cart-code`).

## Goal of the thesis work

Cheaper multi-hop QA: a **small, cheap selector** (few params/FLOPs) decides which retrieved paragraphs the LLM sees, instead of sending all of them. Main benchmark: HotpotQA distractor, the same 400 test questions as the BRACIS paper, plus 1000 dev questions; both are fixed by ID in `experiments/*/split_ids.json`.

## Key files

- `experiments/EXPERIMENTS.md`: **high-level tracker** (status, headline, findings, dated log). Keep it updated.
- `research_direction.md`: options and literature. `bibli.bib` holds all references (arXiv-verified).
- One folder per experiment, each with its own **uv** environment (the user wants each experiment self-contained).

## Results so far

| Exp | Status | Headline |
|---|---|---|
| exp01 hop gate (training-free) | done | 6–10% fewer tokens than Noise-Gate at equal both-gold. Thresholds plateau. |
| exp02 22.7M cross-encoder (`ms-marco-MiniLM-L6-v2`), fine-tuned on a Colab T4 | done | **A1 full fine-tune: gpt-5.4-mini F1 0.796 at 467 tok vs k=10 0.782 at 1448 tok (−68%, paired CI includes 0); top-2 0.780 at 330 tok.** Oracle gold 0.822 (distractors hurt). A2 (n50/n200) and A0 zero-shot are baselines. |
| exp03 LLM weight edit, protocol + probe | probe = **no-go** | The editor-allowed mean-difference direction reproduces A0's score; the best [CLS] linear probe adds only +7 pts top-3, below A2. `PROBE_EXPLAINED.md` explains it simply (the user liked this method; keep it in the dissertation). |
| exp04 Astra edits the selector | **harness built, baselines running, no Astra run yet** | PI override of exp03's no-go (a linear probe can't see non-linear mechanisms). Track C = **compress A1** (remove heads/neurons/layers) vs pruning baselines; Track R = repair A0/A2 (later). Stages: forward-pass budget **B = 5 → 10 → 50**, with gates. |

## exp04 state (`experiments/exp04_astra_edit/`)

- **Committed** (8d1a5c8): `editlib.py` (exact removals, cost), `kernel.py` + `kernel_api.py`, `harness.py`, `prompt_trackC.md` (pre-registered before any run), `baselines.py`, `dummy_script.json`, README, DESIGN.
- **A1 weights:** `models/a1/` (git-ignored; sha256 prefix `605c986eb9a9a82b`). Verified identical to exp02 scores. The original is on Google Drive in `MyDrive/qualiusp2026/exp02_results/full/model`.
- **Dry run** `runs/dry2` passed every check. A1 practice both-gold@3 = 0.945. A full practice pass (1,992 pairs, both models) takes **~5 min on CPU**, so B = 50 means hours of forward time.
- **Running when this was written:** `uv run python baselines.py --eval-n 300` (log `results_baselines_log.txt`, output `results/baselines_dev300.json`). Layer drop, Taylor, Wanda-style and random pruning at FLOPs fractions 0.9–0.5 on 300 dev questions.

### Next steps

1. Read the baseline results; add them to the exp04 README and EXPERIMENTS.md; commit + push.
2. **Stage 1:**
   ```bash
   uv run python harness.py --run s1_C_B5 --track C --budget 5 --turn-cap 40 --editor astra
   ```
   (cwd `experiments/exp04_astra_edit`). Run it in the background. Each turn is one `codex exec` call (~20k tokens of overhead, ChatGPT-plan limits).
3. **Evaluate the final edit:** we still need an `evaluate.py` that builds `el.build_model(models/a1, EditState.load(run/work/edited))`, scores dev (and at stage 3, test), and reports both-gold@3, tokens at the 0.90 threshold, `cost()` and CPU latency. Compare with the baseline curve at the same FLOPs.
4. **Gate:** go to stage 2 (B = 10, turn cap 80) only if stage 1 is valid and shows a change; go to stage 3 (B = 50, turn cap 200) only if Astra ≥ the best baseline at equal FLOPs. Stage 3 adds test + gpt-5.4-mini F1 (`exp02_cross_encoder/generate.py` pattern).

## Operational gotchas

- **GitHub:** push as **arnml**. The active `gh` account sometimes switches to `amoyavt` (push fails with 403). Fix: `gh auth switch --hostname github.com --user arnml`. The user wants everything committed and pushed ("keep all tracked").
- **OpenAI key:** the session env var may be stale. Read it fresh in PowerShell:
  ```powershell
  $env:OPENAI_API_KEY=[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
  ```
  Never print the key. Generator = `gpt-5.4-mini`. Retry/backoff is in `generate.py`; the user worries about rate limits.
- **Codex/Astra:** model `gpt-6-astra` is available in the user's Codex (ChatGPT login). The tool-off flags are in `harness.py` (`--disable` the feature list, `-c mcp_servers.node_repl.enabled=false`, `-c mcp_servers.MCP_DOCKER.enabled=false`, unquoted `-c web_search=disabled`). `resume` must repeat `-m gpt-6-astra`.
- **transformers 5.22 ignores `head_mask`.** Use hooks or editlib for head removal.
- **Colab:** the user's free tier has a T4. Downloads from Drive's web UI don't save from the built-in browser; use a Colab cell with `files.download()` (the user approves the Drive mount popup). Avoid leaving extra Colab sessions open ("too many sessions" error).
- **Windows / Git Bash + PowerShell.** Python is only available via `uv run`. Use `PYTHONIOENCODING=utf-8`.
- **Subagents as LLM generators:** only for tiny pilots (~54k tokens of overhead each). Haiku subagents must write their answer to a file; the hand-back works.

## User preferences

- Straightforward but innovative; honest science: report negative results, pre-register before running, record deviations.
- Explain things simply when asked. Write plain-language md files for methods worth keeping.
- Each experiment self-contained (own dir + uv env). Save results durably (Drive + repo).
- Go step by step with gates (5 → 10 → 50); avoid big bursts of agent or API calls.
