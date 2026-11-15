You are a critical research colleague (ML + statistics + linear algebra) reviewing a thesis project. Read-only: do not modify files, do not run experiments or long commands. Reading files is fine.

Read: experiments/EXPERIMENTS.md, experiments/NEXT_DIRECTIONS.md, and the README.md / PROTOCOL.md of experiments/exp06_skill_size, exp07_chain_selector, exp08_musique, exp09_iterative_rerank, exp10_adaptive_iterative, exp11_iteration_skill (exp11 is running; results so far are below).

Purpose of the thesis: a research contribution (main) with practical value. Setting: a 22.7M MiniLM-L6 cross-encoder used as a cheap evidence selector for multi-hop RAG (HotpotQA, MuSiQue), to cut LLM tokens at a recall target.

exp11 dev300 partial results (recall@5, iterative inference; base = M1 run iteratively 0.45, full = M1-iter 0.74):
rank1 0.663 (74% recovery), rank2 0.727 (95%), rank4 0.730, rank8 0.730, rank16 0.713, random rank-4 same Frobenius norm 0.477 (9%); single layers L0 0.547, L1 0.540, L2 0.537, L3 0.573 (30-42% each). Domain family (A1 -> M1, one-shot; base A1, full M1 0.71): rank1 0.597, rank2 0.647, rank4 0.660.

Give, condensed (tables welcome, max ~150 lines):
1. Critique: weakest claims, statistical problems, confounds (e.g. D_it = M1-iter - M1 mixes separate training runs; recovery on a 300-question set; selection of r* on dev), and what reviewers at ACL/NeurIPS would attack first.
2. Mathematical framing of the low-rank findings. Concrete, computable analyses on the existing deltas (no training) that would turn "rank 2 recovers 95%" into a sharper claim: e.g. singular value spectra / stable rank, principal angles between the top subspaces of D_it, D_dom and the HotpotQA delta (A1 - A0), per-layer subspace overlap, whether the rank-2 directions are shared across layers, linearity of the effect (scaling alpha * delta), interpolation curves. Say which would be most informative and why.
3. Three next experiments ranked by research value x practical value, CPU-feasible (small models, hours not days). For each: hypothesis, design, decision rule, what prior work it must cite or beat.
4. Anything you think we are missing or doing wrong in process (pre-registration, multiple comparisons, reporting).
Be honest and specific; say when something is already known in the literature, citing only papers you are sure exist.
