# Examiner review of the quali draft (Claude, 2026-11-15)

Scope: quali/main.tex, chapters/*.tex, figures/*.tex, checked against HYPOTHESES.md, exp02/06–13 READMEs, bibli.bib, plus 5 papers checked online. I made no edits to the draft. A local latexmk build was not possible because MiKTeX is missing geometry.sty, so section 5 below comes from reading the source only.

## 1. Correctness (ranked by severity)
1. **The "chain adapter" story overreaches what the rank-2 graft gives you** (00_abstract:5, 01:30, 06:9, fig pipeline). Recovery (eq. recovery) is measured from the base model run *iteratively*. That baseline collapses (M1-ctrl iterative 0.4075), so most of the "86.8%" is the model recovering from a collapse, not gaining anything over one-shot. exp12 test: M1-ctrl+rank-2 D, iterative = **0.7025**, vs M1-ctrl one-shot = **0.7075**. So the rank-2 graft keeps *none* of the +0.040 iteration-over-one-shot gain on test. exp11 shows the same pattern: 0.7075 vs M1 one-shot 0.682. Also, the 4-hop confirmation (+0.055) used the full M1-iter, not the graft. The text never says this, yet the abstract writes "rank-2 … and it helps long chains" and the plan says the adapter "turns the one-shot selector into an iterative one".
2. **The non-inferiority rule is stated but not followed** (02:44 vs 05:9, 05:83, abstract:5). 02:44 says "no worse" is "never" claimed from a non-significant difference. Yet the "equal F1" claim (exp02, ΔF1 CIs reaching −0.033) and the exp07/exp10 success rules ("recall CI includes 0") are exactly that. Only exp13 ran a real non-inferiority test. exp02's README also mentions no PROTOCOL, so tab:claims "C = pre-registered" for H1 needs checking.
3. **MuSiQue adaptive stopping is reported selectively** (05:32, tab:claims 05:84, abstract:5). The pre-registered primary was adaptive vs fixed M1-iter: −244 [−296, −192] at 0.95 and **failed at 0.90** (−43 [−95, +9]). The secondary (H8: is the iterative score a better stop signal?) was **refuted**. The text leaves out both. Instead it reports adaptive M1 at "recall 0.95" (it was 0.9475) against sending all 2,535 tokens, which is neither a test nor recall-matched. The status should be "C (HotpotQA); C at 0.95 / N at 0.90 (MuSiQue)". Add H8 = N to tab:claims.
4. **Adaptive-k is misdescribed** (03:8, 05:32). Taguchi et al. (arXiv 2506.08479) pick k at the largest consecutive gap in the top 90% of scores and then **add a buffer B = 5 by default**. So the faithful baseline is the "+5 buffer" row (recall 0.815 vs 0.868), and "3.7 paragraphs, recall 0.50" is a variant without the buffer. Also note that k* = 1 was counted as 2.
5. **The matched-control result is over-read** (05:47). "The result indicates a small effect rather than none" is not supported: CI [−0.005, 0.085], p = 0.11, and the 3 seeds are re-trainings scored on the *same* 400 questions, not replications. The README is more careful ("does not show no effect"). The +0.025 "more training" split also compares a GPU-trained M1-ctrl with exp09's CPU-trained M1, so hardware is confounded.
6. **"All numbers are on held-out questions"** (05:4) is false. 05:42 reports dev300 sweep numbers (0.45/0.74, 74%/95%/97%, random 9%). The random controls (7%, 9%) are dev300 only, from one draw each, and exp11's control is rank 4, not rank 2. The abstract's "≈85% vs 7–9%" therefore sets test numbers against dev300 numbers.
7. **Different models across sections are not disclosed.** sec:res_rank uses exp09's CPU M1/M1-iter (Δ includes 2.2× more training, so it is confounded). sec:res_matched/long use exp12's GPU re-trainings. Say so, and retitle 05:39 "The iteration change (confounded) is rank 2".
8. **"Ranker is the bottleneck" is stated as a finding** (05:34 title, 03:10, 01:29). H5 is E: the primary was not testable, and the k ≤ 20 run was exploratory. Label it as exploratory or descriptive.
9. **Smaller items:**
   - 01:28 claims answer F1 on MuSiQue. None exists yet (exp15 is running).
   - 01:31 says "adversarial review of each protocol". 04:64 says it only started with exp12.
   - 05:32 says "between two and four paragraphs". exp07 reports a mean of 2.02–2.43 from a range of 2–10.
   - 05:32 omits the exp07 F1 follow-up (−0.020 [−0.045, +0.005] at 0.90).
   - 04:10–13: the fresh confirmation set also includes 400 *train-split* 4-hop questions (the stratum with +0.020, n.s.). The data section omits them, and it omits exp08's dev1000.
   - 04:17 says "only the query side is truncated". exp09 reports a longest_first fallback.
   - 06:38 says "single primary per experiment". exp07 and exp10 had multi-target primaries.
   - 04:66 says every citation was verified. bib notes say "Anthology entry not yet checked" (perlman) and taguchi has no URL.
   - 05:72 states "cost-efficient choice" as a conclusion. It rests on descriptive cross-experiment comparisons only.

## 2. Citations
- All 52 \cite keys resolve in bibli.bib, with no duplicate keys.
- Misuse: 02:29 cites vasani2025phlora for the Eckart–Young theorem; cite Eckart & Young (1936) or a textbook. 04:59 uses nosek2018preregistration to justify a protocol "because of AI assistance", which that paper does not discuss. 06:29: braga2025taskarithmetic is *domain* transfer, so cite it as a contrast.
- Spot checks:
  - perlman2026emergence: OK. It studies LoRA on RankLLaMA, and attention updates in a "compact mid-network region" carry much of the gain. Say "LoRA attention updates in the middle of an LLM reranker (correlational)".
  - soligo2025convergent: OK. It uses 9 rank-1 adapters on Qwen2.5-14B, and an *activation* misalignment direction transfers across fine-tunes. Write "activation direction", not "induced direction".
  - morris2026tinylora: OK. 13 parameters, about 90% of the gains under RL, and SFT needs 100–1000× larger updates.
  - altaf2026noisebaseline: OK. Its baseline is a random matrix plus matched fine-tunes, not seed-to-seed deltas, so 06:28 should not attribute the seed baseline to it.
  - taguchi2025adaptivek: the method matches, but the default buffer is missing (see 1.4).

## 3. Originality
No passage reads as a copied abstract. The related-work one-liners are short and in the author's words; 03:22 is closest to the source abstracts, so keep it as is with the fixes above. The boilerplate is 01:6 and 02:8 (hallucination and RLVR with Guo/Lambert/Liu cites). It adds nothing to the argument, so cut it.

## 4. Writing and structure
- Problem, RQs, hypothesis and results are clear. What is missing:
  - (a) an explicit objectives list, as EPUSP quals usually expect;
  - (b) a mapping from RQ1–4 to results sections and to W1–W5;
  - (c) a calendar schedule with dates and the program deadline (06:47 gives quarters only, and "paper submission" names no venue).
- H6-4hop shows heterogeneity: dev +0.139 vs train +0.020 (n.s.). Discuss why. A possible distribution difference between train and dev questions belongs in the threats.
- Undefined or unexplained terms: "SVD" (acronym never introduced), o200k_base (say "the GPT tokenizer"), dev300, "functional rank" (06:28).
- 05:68 has a \todo where a result should be. Either remove sec:res_f1 or state "pending; pre-registered as exp15".
- Candidates to cut for a lean text:
  - 02 §Parametric knowledge, down to 2 sentences;
  - 03:10 and 03:16, which repeat tab:positioning;
  - 04:66 verification item;
  - contribution 4 (01:31), which could become one sentence;
  - unused packages and macros: \argmax, \recallk, ulem, multirow, amsthm, enumerate.

## 5. LaTeX / Overleaf risks
Low risk overall:
- All class macros exist in politex-xico.cls (\capa, \falsafolhaderosto, \qualificacaoMestrado, \areaConcentracao, \data).
- \citeonline is defined in abntex2cite-xico.
- TikZ libraries are loaded, and \resizebox{\input{tikz}} is OK.
- The appendix table has no \caption, which is cosmetic only.
- todonotes renders a box in the final PDF.
- The class re-loads inputenc, which is harmless.
- babel is loaded after abntex2cite. The class reads babel only in \AtBeginDocument, so this is fine.
- I could not compile locally (missing geometry.sty in MiKTeX). Do a first Overleaf build to confirm.

## Suggested rewrites (top 5)
1. Abstract (00:5) and 06:9: "…a rank-2 approximation of this difference restores about 85% of the iterative ranker's recall@5 when run iteratively (random rank-2: 7%, tuning set). It restores the ability to read a conditioned query, but on the test set the grafted model is not yet better than one-shot ranking (0.703 vs 0.708). Separately, the full conditioning-trained ranker improves full-support recall@5 on 566 fresh four-hop questions by 0.055 [0.007, 0.102]. Whether a directly trained rank-2 adapter keeps that gain is the main open question (W1)."
2. 02:44: "Most experiments declared a difference not significant when its CI included 0. This does not establish equivalence. Only the cascade test (exp13) used a pre-set non-inferiority margin, and the headline claims will be re-tested with margins." In tab:claims, change "at equal F1" to "F1 not significantly different (CI ±0.03)".
3. 05:32: "On MuSiQue, the pre-registered test (adaptive vs fixed k on M1-iter) succeeded at recall 0.95 (−244 tokens [−296, −192]) but not at 0.90 (−43 [−95, +9]; τ overshot). The iterative score was not a better stop signal than the one-shot score (+84 tokens at 0.95). Descriptively, adaptive stopping on one-shot M1 reached recall 0.9475 with 1,554 tokens, vs 1,774 for M1 with fixed k."
4. 05:32: "Adaptive-k with its default five-paragraph buffer spent about the same tokens as the learned rule (1,227 vs 1,222) but recovered fewer complete chains (0.815 vs 0.868). Without the buffer it kept 3.7 paragraphs and recall 0.50 (exploratory)."
5. 05:47: "The difference was not significant, and the CI does not exclude a zero or a 0.085 effect (MDE ≈ 0.07). Three seeds gave +0.028 to +0.040 on the same test set (descriptive)."
