# Chain-Following as a Low-Rank Skill

**Cheap evidence selection for multi-hop retrieval-augmented generation**

Master's research at the Escola Politécnica, Universidade de São Paulo (PPGEE, Computer Engineering, PCS).
Author: Arnold Moya Lavado. Advisor: Prof. Dr. Anna Helena Reali Costa.

This repository holds the qualification text, every experiment (code, pre-registered protocol and results), and the review history of both.

## Research question

A retrieval-augmented generation (RAG) system pays for every paragraph it sends to the LLM. This work studies a small selector, a 22.7M-parameter MiniLM cross-encoder, that decides which retrieved paragraphs the LLM reads in multi-hop question answering. The main question is *what such a selector must learn to use the evidence it has already selected, and how that learning is represented in its weights*.

## Main results so far

| Result | Value | Status |
|---|---|---|
| The weight change that lets the selector use already-selected evidence, truncated to rank 2 (SVD), restores its effect on MuSiQue | 86.8% [77.7, 95.8]; random change of equal norm: 7% | confirmed (exp12, secondary test) |
| The same holds on two more datasets | 2WikiMultiHopQA 97.1%, HotpotQA 92.1%; controls ≤ 22% | confirmed (exp16) |
| Iterative selection improves recall@5 on four-hop questions | +0.055 [0.007, 0.102] | confirmed (exp13) |
| The better evidence improves answer F1 on questions with three or more hops | +0.076 [0.041, 0.112] | confirmed (exp15) |
| A router trained on the benefit of iteration improves on one-shot selection | +0.027 [0.016, 0.039] recall@5, with 55% of the passes of always-iterative selection | confirmed (exp14) |
| A small selector reduces prompt tokens on HotpotQA | 68–77% fewer tokens, F1 differences not significant | exploratory (exp02) |

The full claim ledger, including null results, is [`experiments/HYPOTHESES.md`](experiments/HYPOTHESES.md).

## Repository layout

| Path | Contents |
|---|---|
| [`quali/`](quali/) | Qualification text (LaTeX), the source of truth for Overleaf. Includes [`EXPERIMENT_MAP.md`](quali/EXPERIMENT_MAP.md) (which experiments the text uses) and [`REVISION_LOG.md`](quali/REVISION_LOG.md) (how the text was reviewed and revised) |
| [`experiments/`](experiments/) | One folder per experiment, `expNN_name/`, each with its own `uv` environment, `PROTOCOL.md`, code, `README.md` (results) and `results/` |
| [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) | Index and log of all experiments |
| [`experiments/reviews/`](experiments/reviews/) | Adversarial pre-run reviews (Codex), audits of the text, citation checks |
| [`bibli.bib`](bibli.bib) | Master bibliography (copied to `quali/bibli.bib`) |
| [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) | Running notes for continuing the work |
| [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) | Rules for the AI agents that assist this project |
| `quali_main.tex` | Overleaf entry point (inputs `quali/main.tex`) |
| `template.tex`, `politex-xico.cls`, `abntex2*`, `figuras/`, `*.pdf` (root) | The USP LaTeX template these files came from, kept as in the Overleaf project root |
| `references/`, `paperlist.md`, `research_direction.md`, `quali_order.md` | Reading notes and early planning |

## Experiments

| No. | Question | Folder |
|---|---|---|
| 1 | Training-free second-hop threshold on embeddings | [`exp01_hop_gate`](experiments/exp01_hop_gate) |
| 2 | Tokens saved by a fine-tuned 22.7M selector (HotpotQA) | [`exp02_cross_encoder`](experiments/exp02_cross_encoder) |
| 3 | Linear probe for multi-hop relevance | [`exp03_weight_edit`](experiments/exp03_weight_edit) |
| 4 | LLM agents editing the selector's weights | [`exp04_astra_edit`](experiments/exp04_astra_edit) |
| 5 | Which supporting paragraphs the selector misses | [`exp05_failure_anatomy`](experiments/exp05_failure_anatomy) |
| 6 | Is the HotpotQA fine-tuning change low-rank or sparse? | [`exp06_skill_size`](experiments/exp06_skill_size) |
| 7 | Learned adaptive stopping (HotpotQA) | [`exp07_chain_selector`](experiments/exp07_chain_selector) |
| 8 | The HotpotQA selector on MuSiQue | [`exp08_musique`](experiments/exp08_musique) |
| 9 | In-domain and iterative fine-tuning (MuSiQue) | [`exp09_iterative_rerank`](experiments/exp09_iterative_rerank) |
| 10 | Adaptive stopping on the iterative selector | [`exp10_adaptive_iterative`](experiments/exp10_adaptive_iterative) |
| 11 | Is the iteration change low-rank? | [`exp11_iteration_skill`](experiments/exp11_iteration_skill) |
| 12 | Pair-matched control; rank of the conditioning change | [`exp12_matched_control`](experiments/exp12_matched_control) |
| 13 | Four-hop confirmation; hop-count cascade | [`exp13_cascade`](experiments/exp13_cascade) |
| 14 | Router trained on the benefit of iteration | [`exp14_benefit_router`](experiments/exp14_benefit_router) |
| 15 | Answer F1 with the selected evidence | [`exp15_answer_f1`](experiments/exp15_answer_f1) |
| 16 | Rank-2 replication on 2WikiMultiHopQA and HotpotQA | [`exp16_rank2_replication`](experiments/exp16_rank2_replication) |

## Research protocol

- **Pre-registration.** From exp03 onward, each experiment's `PROTOCOL.md` (question, splits, metrics, decision rules, budget, priors) is committed before any evaluation data are scored. Deviations are recorded in the protocol before the affected data are scored.
- **Adversarial review.** From exp12 onward, a second model (OpenAI Codex, `codex exec -s read-only`) argues for the null hypothesis before each run; the reviews are in `experiments/reviews/`.
- **Statistics.** Paired comparisons: exact McNemar tests for per-question outcomes, and paired bootstrap 95% confidence intervals for means and ratios. One primary test per experiment.
- **Reporting.** Null results are reported, and each claim's status is kept in `HYPOTHESES.md`.

## Reproducing an experiment

Each experiment is self-contained:

```bash
cd experiments/exp16_rank2_replication
uv sync
uv run python analyze.py
```

GPU work (training and ranking) runs on a free Colab T4. Each `exp*_colab.ipynb` is generated by `make_notebook.py`, pinned to a commit, and opened from GitHub (`colab.research.google.com/github/arnml/qualiusp2026/blob/<commit>/...`). Inputs are checked by SHA-256 hash. Large artifacts (models, full rankings) stay on Google Drive; the repository keeps the analysis, per-question results and hashes, so the reported numbers can be recomputed. Read each experiment's README for its exact inputs.

Python code is formatted with [ruff](https://docs.astral.sh/ruff/) (configuration in [`ruff.toml`](ruff.toml)).

## Qualification text

The text in [`quali/`](quali/) is compiled only in Overleaf (pdfLaTeX, entry `quali_main.tex`), where each file is linked from this repository's `main` branch. After a change: commit and push, refresh the linked files in Overleaf, then recompile. See [`AGENTS.md`](AGENTS.md).

## Use of AI tools

The project uses AI assistants (Claude Code and OpenAI Codex) for coding, analysis, literature checks and drafting, under the rules in [`AGENTS.md`](AGENTS.md) and [`experiments/AI_USE.md`](experiments/AI_USE.md). Every citation is checked against its official source, and results copied between machines are recomputed.
