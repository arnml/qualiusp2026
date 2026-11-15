## Findings, ranked by severity

### High

1. **“Statistically indistinguishable” is not evidence of equal or non-inferior F1.** [05_results.tex:9](/C:/code/qualiusp2026/quali/chapters/05_results.tex:9), [00_abstract.tex:5](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:5), [01_introduction.tex:28](/C:/code/qualiusp2026/quali/chapters/01_introduction.tex:28)  
   The HotpotQA F1 intervals include zero, but no non-inferiority margin or test is reported for these comparisons. “Matched F1” in the contribution list overstates what the results establish. Rewrite along the lines of: *“At 68–77% fewer prompt tokens, the observed F1 differences were small and their 95% CIs included zero; this does not establish F1 equivalence.”* Remove “matched F1,” or say “compared at similar observed F1.”

2. **The matched-control result is described as evidence of an effect despite an inconclusive primary test.** [05_results.tex:47](/C:/code/qualiusp2026/quali/chapters/05_results.tex:47)  
   The report gives +0.040, CI [−0.005, +0.085], p = 0.109, and says the MDE was about 0.07; it explicitly cautions that this does not show “no effect.” “The result indicates a small effect rather than none” goes further than those data justify. Suggested rewrite: *“The estimate was +0.040, with a 95% CI spanning zero; all three seed estimates were positive, but the primary test did not establish an overall gain.”*

3. **“Separable, low-rank skill” and “rank-2 change” are stronger than the experiment demonstrates.** [00_abstract.tex:5](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:5), [01_introduction.tex:23](/C:/code/qualiusp2026/quali/chapters/01_introduction.tex:23), [01_introduction.tex:30](/C:/code/qualiusp2026/quali/chapters/01_introduction.tex:30), [05_results.tex:56](/C:/code/qualiusp2026/quali/chapters/05_results.tex:56), [05_results.tex:72](/C:/code/qualiusp2026/quali/chapters/05_results.tex:72)  
   The graft result supports *functional recovery under a per-matrix rank-2 approximation*; the reusable adapter has not yet been trained directly, and rank-to-rank differences were a plateau. “Separable” suggests a more established, independently deployable skill. Use: *“The conditioning-related weight difference was functionally approximated by a rank-2 graft, which recovered 86.8% of its measured effect.”* Reserve “adapter” for the planned work.

4. **The abstract presents the MuSiQue stopping result without its exploratory status.** [00_abstract.tex:5](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:5)  
   The report says the pre-registered exp08 comparison was not testable; the k≤20 stopping result is exploratory. Exp10’s 1,554-token adaptive M1 figure is a practical comparison, not a confirmatory test of savings against a matched baseline. Qualify this explicitly: *“An exploratory k≤20 MuSiQue analysis also suggested token savings.”*

5. **“The ranker is the bottleneck” is framed as a conclusion though the planned test was not testable.** [00_abstract.tex:5](/C:/code/qualiusp2026/quali/chapters/00_abstract.tex:5), [05_results.tex:34](/C:/code/qualiusp2026/quali/chapters/05_results.tex:34), [05_results.tex:37](/C:/code/qualiusp2026/quali/chapters/05_results.tex:37)  
   Exp08’s registered stopping comparison failed because A1 could not reach the target within k≤10. The subsequent M1 result is suggestive, but it does not by itself confirm the general bottleneck claim. Prefer: *“The exp08 comparison was not testable at the registered k≤10 range; subsequent in-domain and iterative results suggest ranking quality may be a limiting factor.”*

### Medium

6. **The truncation description omits a documented fallback that changes which side is truncated.** [04_method.tex:17](/C:/code/qualiusp2026/quali/chapters/04_method.tex:17)  
   It says only the query is truncated. Exp09 documents that when a paragraph alone exceeds 512 tokens, truncation falls back to `longest_first`. State the fallback, or narrow the claim to inputs where the paragraph fits within the limit.

7. **“Iteration helps long chains only” overstates the hop-specific evidence.** [05_results.tex:72](/C:/code/qualiusp2026/quali/chapters/05_results.tex:72), [05_results.tex:87](/C:/code/qualiusp2026/quali/chapters/05_results.tex:87)  
   The fresh four-hop primary is confirmatory; the three-hop result is exploratory, and the two-hop comparison is also exploratory. The table correctly labels three-hop as exploratory, but the summary sentence does not. Say: *“Iteration improved the registered four-hop outcome; fresh-set exploratory analyses suggest gains may extend to three-hop questions.”*

8. **The “adaptive stopping saves” summary blends different evidence statuses and baselines.** [05_results.tex:32](/C:/code/qualiusp2026/quali/chapters/05_results.tex:32), [05_results.tex:72](/C:/code/qualiusp2026/quali/chapters/05_results.tex:72)  
   HotpotQA’s savings are registered results. MuSiQue’s 1,554 tokens are adaptive M1 versus sending all paragraphs, while the k≤20 savings are exploratory and partly reflect lower realized recall. Name the baseline and status for each; avoid presenting one general confirmed claim.

9. **Do not describe the four-hop sample as homogeneous.** [05_results.tex:61](/C:/code/qualiusp2026/quali/chapters/05_results.tex:61)  
   The 566 questions combine 166 unused development questions (+0.139) and 400 unused training-split questions (+0.020, nonsignificant). The text gives these strata, but the headline result would benefit from stating that the pooled estimate is heterogeneous. Also clarify that “fresh” means unused questions, not a new dataset or an official test set.

10. **The chapter’s statistical-method summary should distinguish estimation from hypothesis tests.** [02_background.tex:42](/C:/code/qualiusp2026/quali/chapters/02_background.tex:42), [05_results.tex:4](/C:/code/qualiusp2026/quali/chapters/05_results.tex:4)  
    The results chapter says p values come from McNemar “unless stated otherwise,” but the most visible answer-F1 results report bootstrap CIs and no p values. Specify which tests apply to binary recall outcomes and which intervals summarize token/F1 differences.

11. **Methodological claims about pre-registration could be made more precise.** [04_method.tex:62](/C:/code/qualiusp2026/quali/chapters/04_method.tex:62), [04_method.tex:65](/C:/code/qualiusp2026/quali/chapters/04_method.tex:65)  
    “Every experiment” and “exploratory findings must be re-tested … before they are claimed” sound absolute, yet the thesis reports exploratory findings. Consider saying that exploratory results are reported as exploratory and will not be promoted to confirmatory claims without a fresh registered test.

12. **There is no answer-quality result for MuSiQue yet.** [01_introduction.tex:28](/C:/code/qualiusp2026/quali/chapters/01_introduction.tex:28), [05_results.tex:65](/C:/code/qualiusp2026/quali/chapters/05_results.tex:65), [05_results.tex:68](/C:/code/qualiusp2026/quali/chapters/05_results.tex:68)  
    The contribution claims token accounting at matched recall and answer F1 across HotpotQA and MuSiQue, but the MuSiQue answer-F1 experiment is still marked running. Limit the contribution to HotpotQA, and describe MuSiQue answer quality as planned.

13. **The thesis plan schedule lacks calendar anchors.** [06_plan.tex:43](/C:/code/qualiusp2026/quali/chapters/06_plan.tex:43), [06_plan.tex:47](/C:/code/qualiusp2026/quali/chapters/06_plan.tex:47)  
    “Q1–Q4 from the qualification” is hard to assess without the qualification date or expected defence date. Give month/year ranges or state the expected qualification month.

### Lower severity: citations, originality, and concision

14. **Citation-key check:** I found no `\cite` keys missing from `quali/bibli.bib`. The PHLoRA citation appears to support the stated truncated-SVD method; the official paper describes post-hoc extraction from weight deltas using truncated SVD ([PHLoRA paper](https://aclanthology.org/2025.findings-ijcnlp.125/)). Noise-Gate is cited as the author’s earlier work and the Springer record confirms the title and authors ([Springer proceedings](https://link.springer.com/book/10.1007/978-3-032-39892-5?page=2)).

15. **Originality/self-plagiarism:** I found no passage in the reviewed text that clearly looks copied from a known abstract. Reuse of the BRACIS HotpotQA test set and prompt is disclosed in [04_method.tex:6](/C:/code/qualiusp2026/quali/chapters/04_method.tex:6) and [05_results.tex:9](/C:/code/qualiusp2026/quali/chapters/05_results.tex:9). That is good provenance practice; it does not establish whether any wording or figures were reused, which would require comparing the actual BRACIS manuscript.

16. **Lean-text opportunity:** the five-item research-protocol list is unusually prominent relative to the qualification’s research question and results. [04_method.tex:56](/C:/code/qualiusp2026/quali/chapters/04_method.tex:56)–[04_method.tex:66](/C:/code/qualiusp2026/quali/chapters/04_method.tex:66)  
    Condense it to a short reproducibility statement and move operational details (AI review, checksum transfer checks, claim ledger) to an appendix or repository documentation.

17. **Visible encoding corruption in document metadata.** [main.tex:26](/C:/code/qualiusp2026/quali/main.tex:26), [main.tex:28](/C:/code/qualiusp2026/quali/main.tex:28)  
    The source contains `ComputaÃ§Ã£o` and `SÃ£o Paulo`. These should be corrected to Portuguese UTF-8 text before producing the qualification PDF.