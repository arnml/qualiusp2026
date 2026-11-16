# Revision log of the qualification text

How the text in `quali/` was written, reviewed and revised. It is compiled only in Overleaf (project `qualiusp2026`, entry `quali_main.tex`), from the files linked to this folder. The full history is in `git log -- quali`. The review reports named below are in `experiments/reviews/`.

## Round 1, 2026-11-15: first version

- First full draft in English. It covers the experiments relevant to the deliverable; all experiments are tracked in `EXPERIMENT_MAP.md`.
- Audits:
  - Codex: correctness, plagiarism and academic writing (`codex_quali_review.md`, `codex_quali2_review.md`).
  - Claude reviewer (`claude_quali_review.md`).
- Main fixes:
  - It is now disclosed that the graft did not beat one-shot selection overall.
  - The Adaptive-k buffer and the exp10 null at 0.90 are now reported.
  - Uncited claims were either cited or softened.
- Research-first framing; hypothesis figure added to the Introduction.

## Round 2, 2026-11-15: new results and sources

- **exp16 replication** added (Section 5.6): rank-2 recovery of 97.1% on 2WikiMultiHopQA and 92.1% on HotpotQA.
- **exp14 benefit router** added (Section 5.9): +0.027 recall@5 over one-shot, with 55% of the passes of always-iterative.
- **Bibliography** moved to published versions: 35 preprints replaced, 3 errors fixed (`BIB_PUBLISHED_VERSIONS.md`).
- **Citation check against the papers:** 10 overstated attributions fixed; Nijasure et al. 2025, Lu et al. 2025 and Shi et al. 2023 added (`citation_check_2026-11-15.md`).
- **Codex examiner review:** no numeric errors; claims scoped to the evidence (`codex_quali3_examiner.md`).

## Round 3, 2026-11-16: research tone, clarity and formal method

Three independent audits, each from its own point of view:

| Audit | File |
|---|---|
| Scientific tone | `audit_tone_2026-11-16.md` |
| Clarity for the advisor | `audit_clarity_2026-11-16.md` |
| Committee examiner | `audit_examiner_2026-11-16.md` |

Changes:

- **Abstract and resumo:** lead with the research result (functional rank of the conditioning change), give exact parameter counts, and present the cost results as a consequence.
- **Introduction:** rebuilt as context, research gap, RQ1–RQ4 (RQ4 is the main question), hypothesis, contributions tagged with RQ and section, and organization.
- **Background:** a formal SVD subsection:
  - decomposition, norms and truncated SVD in LoRA form;
  - the Eckart–Young theorem (Eckart & Young 1936; Mirsky 1960; Golub & Van Loan 2013);
  - energy and stable rank.
- **Method:**
  - definitions of graft, recovery and functional rank;
  - seed-noise and random controls;
  - cascade, router and oracle;
  - a "Deviations" paragraph.
- **Results:**
  - reordered by research question;
  - a rank-1 column in the replication table;
  - a separate cascade section and table (the two question sets are kept apart);
  - a claims table with an RQ column and the main claim first.
- **Plan:**
  - follow-up experiments W1–W5 and a locked-set confirmation, each with question, design, decision rule and expected outcome;
  - expanded threats to validity: multiplicity, single seeds, one-dimensional changes, truncation, evaluation data, construct validity, reader and sequential design.
- **Figures:** routing is labelled by predicted benefit, not hop count.
- **Appendix:** Experiments 1–16 written as question, answer and status.
- **Bibliography:**
  - Recheck of preprints: MS MARCO is now cited from the CEUR-WS proceedings.
  - Peer-reviewed sources added: Dai & Callan (SIGIR 2019) and FIRST (EMNLP 2024).
  - The ABNT alphabetical order is kept, since the first and last entries are peer-reviewed.

### Full-document consistency audit (`audit_full_consistency_2026-11-16.md`)

Every repeated number matches the results files. The following were fixed:

1. **Main result's status.** The 86.8% recovery was exp12's pre-registered *secondary* test; it is now labelled that way.
2. **Functional rank 1** on 2Wiki and HotpotQA is reported as tuning-set only, not yet confirmed. The confirmed claim is "functionally rank 2 on three datasets".
3. **Pre-registration.** It applies from exp03 onward. exp02's token result is marked exploratory (no protocol file), here and in `experiments/HYPOTHESES.md` (H1).
4. **Codex pre-run reviews** apply from exp12 onward.
5. **M1-iter** was trained twice with the same recipe (0.750 and 0.7475); both trainings are now named.
6. **Related work vs Section 5.4:** the statement about the layer profile now agrees with Section 5.4.
7. **Hypothesis figure:** panel (b) now shows the router; the matrix vs whole-model parameter counts are explicit.
8. **The hypothesis** is stated identically in the Introduction and the Plan: a benefit on a subset of questions, identifiable before iteration is run.
9. **Seed-noise control:** it was run on the two replication datasets only, and the text now says so.
10. **Smaller fixes:** the cost sentence in the summary, status words in the appendix, F1 wording in the abstract, router inputs, threats about evaluation data, and the rank set {1, 2, 4, 8}.

## Round 4, 2026-11-16: repository link and code cleanup

- **Repository link.** The public link `https://github.com/arnml/qualiusp2026` now appears in the Introduction (Organization), the research protocol (Section 4.6) and the experiment index.
- **Root README.** A new `README.md` describes the research question, main results, repository layout, the experiment list, the research protocol and how to reproduce an experiment.
- **Code cleanup** (67 project files in `experiments/`; behaviour unchanged):
  - formatting with ruff (line length 120; configuration in `ruff.toml`);
  - imports sorted and 5 unused imports removed;
  - `lru_cache(maxsize=None)` replaced by its equivalent `cache`;
  - constant word lists written as list literals.
- **Excluded from the cleanup:**
  - the recorded LLM-agent code in `exp04_astra_edit/runs/` (research data, kept verbatim);
  - the generated Colab notebooks.
- **How behaviour was checked:**
  - Every file's syntax tree was compared with the previous commit, ignoring formatting, comments and docstrings. The only differences are the equivalent rewrites listed above.
  - All files compile.
  - The exp14 analysis was re-run from the rankings stored in the repository and compared with its saved output.

## Round 5, 2026-11-16: figures

- **Figure 1** keeps only the routing panel: one-shot selector, router, and either one-shot selection or the rank-2 adapter with iterative selection. The parameter counts it used to show are in the abstract and Section 2.5.
- **New Figure (Section 5.4): the MiniLM-L6 cross-encoder.**
  - It shows where the graft changes the weights: the four attention matrices and two feed-forward matrices of every layer, the embeddings, and the biases and normalization in full.
  - It shows the single-layer recoveries of exp11: L0 33%, L1 31%, L2 30%, L3 43%, L4 40%, L5 8%.
  - The values were checked against `experiments/exp11_iteration_skill/results/results.json`. A draft note claimed L2 = 44%; the results file does not support it.
- **Section 5.4 text** now says that:
  - the layer grafts add a layer's full change and were measured on the tuning set;
  - the single-layer recoveries overlap (they sum to about 180%);
  - attention and feed-forward have not been separated.
- **Follow-up W4** now includes attention-only, feed-forward-only and single-layer grafts of the conditioning change.

## Open items for the author

- Optional: give the schedule in calendar months once the exam date is set (the in-text reminder was removed on 2026-11-16).
- Review the Portuguese resumo.
- Optional: Portuguese hyphenation for the resumo (`babel` with `portuguese`), if the class does not already set it.
