# exp14: a cascade with a router trained on iteration benefit

**Question.** Can a cheap router, trained on per-question benefit (iterative hit@5 − one-shot hit@5) from one-shot signals, capture iteration's recall gain at a fraction of its cost?
Protocol: [`PROTOCOL.md`](PROTOCOL.md) (pre-registered at `11296db`; Codex review `../reviews/codex_exp14_review.md`; **deviation 1** before any ranking, Codex review `../reviews/codex_exp14_deviation1.md`).

**Run.** Colab free T4 on the second Google account, notebook [`exp14_colab.ipynb`](exp14_colab.ipynb) at `7053e2c`, code pinned to `1742d41`. torch 2.14.1+cu130, transformers 5.22.1.

## Deviation 1: the models were re-trained

The first account's Drive, which held the exp12 models and the exp13 rankings, was not reachable. M1-ctrl and M1-iter (seed 0) were re-trained with exp12's code at its registration commit (`975371a`), using the same data (sha-checked), seed and settings.

- **Identity.** The weight files differ from the registered ones (new sha prefixes `e59ee318…` and `13649d11…`), so these are re-trainings, not the exact exp12 models.
- **Fidelity** (exp12's 400-question test set):
  - iter_iter 0.7475 and ctrl_oneshot 0.7075 recall@5, exactly as in exp12;
  - **0 of 400 questions change their hit@5** for either model.
- **Rankings.** All rankings were made with the re-trained models, including `router_train` and `dev_fresh`. Both the notebook and `analyze.py` assert these hashes (`results/model_sha.json`).

## Results (`b_eval`, n = 2,000 unused MuSiQue train questions; registered run = Colab)

| | recall@5 | passes / question |
|---|---|---|
| One-shot (M1-ctrl) | 0.8445 | 20 |
| Always iterative (M1-iter) | 0.8665 | 74 |
| **Benefit cascade** (28% routed) | **0.8715** | **40.8** |
| Hop-count cascade (exp13 router, refit) | 0.8425 | 28.7 |
| Oracle benefit router (upper bound) | 0.9395 | 27.0 |

- **Gate:** iterative − one-shot = +0.022 [0.004, 0.040] ≥ 0.02 ✅
- **Primary:** benefit cascade vs one-shot **+0.027 [0.016, 0.039]**, McNemar 96/42, p = 5 × 10⁻⁶; passes 40.8 ≤ 50; effect ≥ +0.01. **Holds.**
- **Secondary (exploratory):**
  - **a.** Beats the hop router: +0.029 [0.016, 0.042], p = 1 × 10⁻⁵.
  - **b.** Non-inferior to always-iterative: +0.005, one-sided 95% lower bound −0.0065 > −0.01 ✅ (fragile; see the robustness note).
- **Descriptive:**
  - Router AUC for "iteration helps" 0.598; a text-only router reaches 0.551.
  - τ bootstrap on `b_tune`: routed fraction 0.24–0.40.
  - 2×2 recall@5: ctrl_iter 0.546 (the one-shot model collapses when run iteratively); iter_oneshot 0.810.
  - Primary difference by truncation: −0.011 when the iterative query reaches 512 tokens (n = 282), +0.033 otherwise.
  - Recall@5 by hops, cascade: 2-hop 0.912, 3-hop 0.771, 4-hop 0.556.
  - `dev_fresh` (re-used set): one-shot 0.693, iterative 0.745 (the same as exp13), cascade 0.719 with 42 passes (p = 0.010 vs one-shot).

## Robustness of the router fit

Re-running `analyze.py` locally on the same rankings (scikit-learn 1.9.1, Windows) gives a slightly different router, τ and routed set ([`results_local_rerun_sklearn1.9.1.json`](results/results_local_rerun_sklearn1.9.1.json)):

- **Primary:** still holds, +0.023 [0.012, 0.035], p = 1 × 10⁻⁴, 40.0 passes.
- **Secondary a:** still holds (+0.025, p = 2 × 10⁻⁴).
- **Secondary b:** the non-inferiority lower bound is −0.0105, just past the margin. **Non-inferiority to always-iterative is therefore not robust;** a fair summary is that the cascade matches always-iterative within about ±0.01 at 55% of its passes.

The `HistGradientBoostingRegressor` fit is not bit-reproducible across environments. The Colab run is the registered one, but the protocol should have pinned scikit-learn.

## Interpretation

Iteration's benefit is weakly predictable from one-shot signals (AUC 0.60), and that is enough: routing 28% of questions recovers all of iteration's gain on this train-distribution sample at 55% of its cost. The oracle (0.94 at 27 passes) shows large headroom for a better benefit predictor. The sample is mostly two-hop (natural mix), so the absolute gain is small. Unlike exp13's hop router, this router targets benefit, and it wins.

## Files

- `results/results.json`: the registered analysis (Colab).
- `results/results_local_rerun_sklearn1.9.1.json`: the local re-run.
- `results/rank_*.jsonl.gz`: per-question orderings, passes, top-10 scores and model hashes for every split and config; `analyze.py` runs on them once they are un-gzipped.
- `results/model_sha.json`: the re-trained model hashes.
- `results/fidelity/eval_test_exp12.jsonl.gz`: the fidelity check.
- Models are on the second account's Drive (`MyDrive/qualiusp2026/exp14_1742d41/models`).
