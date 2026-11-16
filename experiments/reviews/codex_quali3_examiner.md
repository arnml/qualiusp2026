### Findings

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:18) | “The cost of RAG can be reduced without losing answer quality” generalizes beyond the tested settings; HotpotQA F1 differences were inconclusive without a non-inferiority test. | Limit this to the measured selector, datasets, and reader, and describe the F1 result as inconclusive on HotpotQA.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:7) | The Portuguese summary makes the same broad cost-without-quality-loss claim. | Apply the same qualification as in the English abstract.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:16) | “The weight change that gives the selector this ability” and “the model cannot use” imply a causal mechanism. The graft results show functional recovery for this setup, while matched-data iteration’s overall primary comparison was not significant. | Say the rank-2 graft recovers much of the measured conditioning-change effect; avoid claiming it establishes the mechanism.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:5) | Same causal wording appears in Portuguese: the change “dá ao seletor essa capacidade” and the model “não consegue usar” chosen paragraphs without it. | Recast as a functional graft result, scoped to the tested model and data.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:114) | “A router trained on benefit recovers iteration’s gain” can read as matching the always-iterative system; that non-inferiority finding was exploratory and failed on the local rerun. | State that the cascade’s registered primary beat one-shot on the evaluation sample; describe matching always-iterative as fragile/exploratory.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:16) | “Recovered that gain” may imply the benefit router reliably preserves the always-iterative gain. Its primary comparison was against one-shot; non-inferiority to always-iterative was fragile and failed robustness. | Specify that it improved over one-shot by 0.027 on the sampled train-distribution evaluation, using 55% of always-iterative passes; qualify the comparison to always-iterative.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:18) | It says the remaining work will combine the adapter with “this router,” although the benefit router is not robustly non-inferior to always-iterative and the adapter has not yet been trained directly. | Present adapter-router integration as planned work whose effectiveness remains to be established.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:110) | “Functionally rank 2 on three multi-hop datasets” overstates the replication slightly: rank 2 met the preregistered recovery rule, but rank 1 already recovered 84–91%, and controls were single tuning-set draws. | Say the rank-2 recovery criterion passed on three datasets, with rank 1 already recovering most of the effect and controls limited to single draws.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:92) | Replication is described as repeating the “matched-control design,” but the protocol’s decision is about rank-2 recovery, with rank/noise controls on tuning data; the cross-dataset test does not establish iteration’s benefit. | Call it a replication of the rank-2 recovery analysis and keep the distinction from benefit explicit.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:65) | “reproduced the earlier test results question by question” is broad: the reported fidelity check establishes unchanged hit@5 on the 400 questions, not identity of all test outputs or results. | Narrow this to “reproduced hit@5 on all 400 test questions.”

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:61) | The four-hop primary combines 166 unused development and 400 unused training questions; the strata are heterogeneous, with the training-source difference only +0.020 and not significant. Calling the aggregate “confirmed” elsewhere risks obscuring that limitation. | Keep the source heterogeneity adjacent to any claim that the four-hop effect is confirmed.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:65) | “an oracle router would reach 0.94” could be read as a validated system result; it is an upper-bound diagnostic, not a practical router. | Label it explicitly as an oracle upper bound.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:37) | “ranking quality is the main limiting factor” is stronger than the evidence: exp08’s primary was not testable, and later results only suggest this interpretation. | Keep “suggest” and scope the claim to these MuSiQue results.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:70) | “On two-hop questions it was non-inferior” is immediately labeled exploratory, but the claims table omits this caveat by saying only “Iteration helps three-hop chains, not two-hop.” | Make the claims-table entry explicitly exploratory for both the three-hop gain and two-hop non-inferiority.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:132) | “The long-chain recall gain improves answer F1” reads as a general causal claim; the registered primary was a subgroup chosen from recall results on the same questions. | State the observed +0.076 F1 result on the selected ≥3-hop subgroup and retain the selection caveat in the table.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:110) | “rank 1 already recovered 84–91%” gives a range without saying these are the dataset-specific test estimates; the main table shows only rank-2 recovery. | Label the figures as rank-1 test recovery for 2Wiki and HotpotQA.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:4) | “All numbers are on held-out questions” conflicts with tuning-set numbers reported later, including rank sweeps and single-draw controls. | Say primary test estimates use held-out questions; identify tuning-set analyses where they appear.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:16) | “random changes … 7–9%” sounds like a consistent control range, but the replication controls vary substantially: 3% on 2Wiki and 22% on HotpotQA. | Scope 7–9% to the MuSiQue experiments; report the replication controls separately or omit the range.

- [05_results.tex](/C:/code/qualiusp2026/quali/chapters/05_results.tex:110) | Says rank-2 recovery “stayed high at recall@5” without giving the dataset-specific values; the JSON reports 97.2% on 2Wiki and 97.9% on HotpotQA. | Give the values and specify their datasets if retained.

- [00_abstract.tex](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:7) | The Portuguese summary says the rank-2 result “esclarece o que o ajuste fino ensina”; this suggests an explanatory conclusion beyond the functional recovery evidence. | Say the result characterizes the tested weight change, rather than explaining fine-tuning generally.
### Response (applied 2026-11-15)

No numeric mismatches were found. Changes made:

- **Abstract and resumo:** rewritten to describe the result functionally, not causally:
  - "the weight change learned by iterative training" replaces "the change that gives this ability"
  - "the one-shot model run iteratively loses most of its coverage" replaces "cannot use"
  - the 7–9% random controls are scoped to MuSiQue, and the replication's controls (3% and 22%) are added
  - the router result is now stated as its registered comparison: +0.027 [0.016, 0.039] vs one-shot, 28% routed, 55% of iterative passes
  - the cost claim is limited to the tested datasets and reader, with F1 "not measurably lower (HotpotQA) or higher (MuSiQue)"
  - combining the adapter with the router is now framed as remaining work
- **Results:**
  - Held-out vs tuning-set numbers are now distinguished.
  - The replication is called a replication of the rank-2 analysis.
  - The rank-1 and recall@5 values are given per dataset.
  - The text now says "the rank-2 criterion passed", with rank 2 read as an upper bound on the functional rank and the controls described as single tuning-set draws.
  - Fidelity is narrowed to hit@5 on the 400 questions, and the oracle is labelled an upper bound.
  - The summary and claims table are qualified: four-hop heterogeneity by source, the two-hop result marked exploratory, F1 on a subgroup chosen from recall, and the router claim stated against one-shot.
- **Introduction and plan:** the router claims are aligned with the above.
