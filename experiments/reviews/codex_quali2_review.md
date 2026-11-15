## 1. Uncited strong claims and result strength

Ranked by risk of misleading the reader.

1. **[05_results.tex:100] “Small selector: 68--77% fewer tokens at equal F1” is too strong.** The text at [05_results.tex:9] says the F1 confidence intervals include zero, but also says no non-inferiority margin was registered. That does not establish equal F1. **Fix:** “Small selector: 68--77% fewer tokens; F1 differences were small and not statistically significant.”

2. **[05_results.tex:9] “distractors actively hurt the reader” overstates what the comparison establishes.** Gold paragraphs change both relevance and context length, so this is not an isolated test of distractor harm. **Fix:** “The gold-paragraphs condition had 0.040 higher F1 than \(k=10\) (95% CI [0.017, 0.064]); this is consistent with irrelevant context reducing answer quality.”

3. **[03_related_work.tex:10] “matched recall and answer quality” implies equivalence.** The HotpotQA results do not establish equivalent answer quality. **Fix:** “cost accounting at specified recall targets, with answer-F1 differences and confidence intervals reported.”

4. **[01_introduction.tex:8] “A selector that scores each passage against the question alone tends to rank such bridge passages low” is a factual claim without a citation or supporting result in the main text.** **(c)** Soften to “can rank such bridge passages low,” or report the relevant experiment and cite it.

5. **[02_background.tex:18] “It is more accurate” (cross-encoders versus bi-encoders) lacks a direct citation.** **(a)** Add the existing `\cite{nogueira2019passage}` to that sentence, or **(c)** soften to “can model interactions between the question and paragraph, at the cost of one forward pass per pair.”

6. **[04_method.tex:6] HotpotQA’s distractor setting and its “two supporting” paragraphs are uncited.** **(a)** Add `\cite{yang2018hotpotqa}` to the dataset description.

7. **[01_introduction.tex:6] “This makes answers easier to update and to verify” is uncited at the point of the claim.** **(a)** Attach the existing RAG citation `\cite{3495724.3496517}` to that sentence.

8. **[03_related_work.tex:36--38] The “Known” cells list Adaptive-\(k\), DynamicRAG, Self-RAG, Beam Retrieval, MDR, Baleen, LoRA, and PHLoRA without citations in the table.** The prose above cites these works, but the table should stand on its own. **Fix:** Add the relevant existing keys to each cell: `taguchi2025adaptivek`, `sun2025dynamicrag`, `asai2024selfrag`; `zhang2024beamretrieval`, `xiong2021mdr`, `khattab2021baleen`; and `hu2022lora`, `vasani2025phlora`.

Other result wording to tighten: **[05_results.tex:37]** says M1-iter “reached 0.750” after “raised” to 0.682; the comparison is confounded by training-data volume. Say “The un-matched M1-iter model reached 0.750; this comparison is confounded…”. **[05_results.tex:56]** “\(\Delta\) encodes the ability to read a conditioned query” is stronger than a functional graft result. Say “The result supports the interpretation that \(\Delta\) helps the model use a conditioned query.” **[05_results.tex:89]** calls the selector “the cost-efficient choice” without qualifying the tested datasets, reader, or compute budget. Limit the claim to those conditions.

## 2. Purpose and relevance

- **Abstract [00_abstract.tex:10--14] and introduction [01_introduction.tex:6--12]** make the token-cost and multi-hop problem clear. The scientific question—what fine-tuning changes—is clearest in [01_introduction.tex:18, 23]. Add one sentence to the introduction tying the two stakes together: *“The practical question is whether a small selector can reduce reader context cost without losing evidence; the scientific question is what change lets it follow evidence across hops.”*
- **Background [02_background.tex:4]** lists topics but not why they matter. Add: *“These concepts define the cost, evidence, and model changes measured in the experiments.”*
- **Related Work [03_related_work.tex:4]** says it positions the work, but delays the gap until [03_related_work.tex:10, 16, 24]. Add an opening sentence: *“Existing work reduces context or retrieves iteratively, but leaves open whether a small reranker can follow evidence chains and whether that behavior can be isolated in its weights.”*
- **Materials and Methods [04_method.tex:1--4]** has no chapter-opening rationale; it starts directly with data. Add: *“This chapter specifies the controlled comparisons needed to separate token savings, iterative conditioning, and training-data effects.”*
- **Preliminary Results [05_results.tex:4]** opens with test procedures, not the reader’s question. Add: *“The results first measure context savings, then test whether conditioning helps multi-hop evidence selection and answer quality.”*
- **Research Plan [06_plan.tex:6--13]** describes an artifact, but the deployment value is buried in line 13. Lead with: *“The remaining work tests whether one small selector can serve short questions cheaply and activate iterative selection only when it improves long-chain evidence.”*

## 3. Writing style: eight priority rewrites

1. **[03_related_work.tex:8]** The long catalogue is hard to scan. Split after the compression methods and adaptive methods; give each sentence one category and a short comparison.
2. **[05_results.tex:32]** Split the sentence into HotpotQA findings, MuSiQue findings, and the Adaptive-\(k\) comparison. Define “complete chains” as “all annotated supporting paragraphs” on first use.
3. **[05_results.tex:42]** Too many results and comparisons are packed together. Separate the rank-2 result, random control, layer analysis, and plateau interpretation into four sentences.
4. **[05_results.tex:56]** Define “recovery” in plain language here, since readers encounter it earlier in the abstract: *“Recovery is the fraction of the performance gap from the collapsed iterative baseline to M1-iter that the graft closes.”* Replace “encodes the ability” as noted above.
5. **[05_results.tex:61]** “Fresh” needs repeated qualification, and the sentence mixes sources, heterogeneity, seeds, and exploratory subgroup results. Report the registered four-hop test first; move the exploratory development-set pattern to a separate sentence.
6. **[05_results.tex:68]** The reader setup, primary subgroup, all-question result, two-hop result, token comparison, and selection caveat make this paragraph dense. Separate the primary finding from exploratory results and the subgroup-selection caveat.
7. **[00_abstract.tex:12]** “pair-matched,” “graft,” and “recovery” appear before explanation. Replace with plain language at first mention, e.g. *“a control trained on the same examples”* and *“a rank-2 weight update restored 86.8% of the measured performance gap.”*
8. **[02_background.tex:34]** “Full-support recall” is defined, but “matches what a reader needs to answer” overgeneralizes because readers may answer without every annotated support paragraph. Replace with: *“This measures whether the selected set contains every annotated support paragraph; it is stricter than per-paragraph recall.”*

Across the text, use **“supporting paragraphs”** consistently instead of alternating among “evidence,” “support,” and “chain” when referring to annotated paragraphs. Define **“conditioning”** on first use as adding already-selected paragraphs to the query, and use **“rank-2 graft”** consistently for the extracted update.