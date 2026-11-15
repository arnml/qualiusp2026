# Citation check, 2026-11-15

Scope: every `\cite` / `\citeonline` in `quali/chapters/*.tex` (29 citing lines, about 50 keys), checked against the paper's arXiv abstract or HTML (full text where it mattered), the ACL Anthology, or a web-search summary. Nothing in the repo was edited. LaTeX was not run.

Legend. OK = supported. OK(abs) = supported by the abstract only, full text not opened. NO = not opened this session (well-known paper, claim is generic); counted as OK in the totals but not independently re-verified here.

Source of venue/year: `quali/bibli.bib` and `experiments/reviews/BIB_PUBLISHED_VERSIONS.md`.

## Table

| file:line | cited key | our claim (short) | verdict | evidence + source | suggested fix |
|---|---|---|---|---|---|
| 03:8 | xu2024recomp | RECOMP trains compressors that summarize retrieved docs | OK | "extractive ... selects useful sentences"; "abstractive ... generates summaries" https://arxiv.org/abs/2310.04408 | none |
| 03:8 | jiang2024longllmlingua | prunes tokens | OK | "a token-level subsequence of" the prompt https://arxiv.org/html/2310.06839 | none |
| 03:8 | wang2023filco | filters context at span level | overstated | FilCo filters "in a fine-grained sentence-wise manner" https://arxiv.org/html/2311.08377 | "at the sentence level" |
| 03:8 | chirkova2025provence | prunes sentences with a trained cross-encoder | OK | sentence-level pruning, DeBERTa-v3 cross-encoder reranker, per-token labels https://arxiv.org/html/2501.16214 | optional: add "and reranks in the same pass" |
| 03:8 | yoon2024compact | compresses iteratively | OK | "iterative architecture where the compressed contexts are updated at each iteration" https://arxiv.org/html/2407.09014 | none |
| 03:10 | asai2024selfrag, jeong2024adaptiverag | LLM or classifier decides whether/how much to retrieve | OK (NO) | not re-opened | none |
| 03:10 | sun2025dynamicrag | learns the number of documents | OK | reranker trained by RL from generator feedback, picks a per-query document count https://arxiv.org/abs/2505.07233 | none |
| 03:10, 05:32 | taguchi2025adaptivek | cuts at largest gap, buffer of 5 default | OK | "choose the index k where the similarity drop is the largest"; "we set B=5" (Sec. 3.3) https://arxiv.org/html/2506.08479v3 | Add that the gap search is limited to the top 90% of documents, and that the paper tests only bi-encoder cosine scores (Contriever, BGE, GTE), not cross-encoders. The 05:32 wording ("applied to the same cross-encoder scores") is already correct. |
| 03:10 | lavado2027noisegate (self) | 26-27% fewer tokens on HotpotQA; Adaptive-k selected too few and lost quality | OK (internal) | repo notes: F1 0.627 vs 0.651 / 0.716 / 0.792 for k = 3 / 5 / 10 (`experiments/exp10_adaptive_iterative/README.md:83`); 26.7% in EXPERIMENTS.md. Paper itself not opened (Springer). | Say "lost answer quality (below fixed k = 3)" to match the numbers. Year: key and bib say 2027 (print year) while the venue is BRACIS 2026; fine, but the in-text label will read 2027. |
| 03:12 | sun2023rankgpt, pradeep2023rankzephyr | listwise LLM rerankers accurate but costlier | OK (NO) | generic | none |
| 03:16 | xiong2021mdr, khattab2021baleen | retrieve iteratively, conditioning on what was found | OK (NO) | generic | none |
| 03:16 | trivedi2023ircot | interleaves retrieval with CoT by an LLM | OK (NO) | generic | none |
| 03:16 | zhuang2024efficientrag | replaces some LLM calls with small models | OK(abs) | "iteratively generates new queries without the need for LLM calls at each iteration" https://arxiv.org/abs/2408.04259 | none |
| 03:16 | zhang2024beamretrieval | cross-encoder scores candidate + question + chosen passages, beam search; DeBERTa base and large | OK | "We employ the base and the large version of DeBERTa"; later hops score question + partial hypothesis + candidate https://arxiv.org/html/2308.08973 | Detail: it is one encoder with two heads (hop 1, later hops). Not an issue for the text. |
| 03:16 | zhang2024beamretrieval | "reports high support recall on MuSiQue" | overstated | The paper reports retrieval EM/F1, not recall: dev EM 77.4/79.3, F1 89.8/90.5 (beam 1/2); test Sp F1 90.0/91.4 https://arxiv.org/html/2308.08973 | "reports about 0.90 support-passage F1 on MuSiQue-Ans" (or "high retrieval F1") |
| 03:22 | li2018intrinsic, aghajanyan2021intrinsic, hu2022lora | low intrinsic dimension; LoRA works | OK (NO) | generic; li2018 and hu2022 venues rest on arXiv comments, not re-verified (see BIB_PUBLISHED_VERSIONS) | none |
| 03:22, 02:32 | vasani2025phlora | extracts low-rank adapters from full fine-tuning checkpoints with truncated SVD | OK | "weight difference ... truncated SVD ... best rank-r approximation" https://arxiv.org/pdf/2509.10971 ; Findings IJCNLP-AACL 2025 matches bib | none. Note it was evaluated on Amazon Nova LLMs, not small encoders. |
| 03:22 | panigrahi2023skilllocalization | skills concentrate in few parameters or layers | overstated | Abstract: about 0.01% of parameters give over 95% of fine-tuned performance; abstract says nothing about layers https://arxiv.org/abs/2302.06600 | "in a few parameters" (drop "or layers") |
| 03:22 | perlman2026emergence | middle layers of a large reranker | overstated (minor) | RankLLaMA; "restricting LoRA attention updates to a compact mid-network region is sufficient for recovering over half"; authors call it correlational https://aclanthology.org/2026.findings-emnlp.1948/ | "restricting LoRA attention updates to a mid-network region recovers over half of the gain in RankLLaMA" |
| 03:22, 06:25 | biderman2024loralearnsless, shuttleworth2025illusion | LoRA updates differ structurally from full FT | OK | full FT perturbations have rank 10-100x typical LoRA (Biderman, https://arxiv.org/abs/2405.09673); LoRA has "intruder dimensions" (Shuttleworth, https://arxiv.org/abs/2410.21228) | none. bib issue: shuttleworth2025illusion has year 2026 though URL is NeurIPS 2025 (vol. 38); see Venue notes. |
| 03:22, 06:28 | altaf2026noisebaseline | rank measures need a noise baseline | OK | baseline = "random matrix of the same shape", plus pretrained weights and matched fine-tunes https://arxiv.org/abs/2610.22438 | In 06:28 "seed-to-seed differences" is our own addition; fine, but do not attribute it to Altaf. |
| 03:24 | turner2025modelorganisms, soligo2025convergent | rank-1 adapters suffice for emergent misalignment; induced direction transfers across fine-tunes | overstated (minor) | Turner: "induce misalignment using a single rank-1 LoRA adapter" (https://arxiv.org/abs/2506.11613). Soligo: 9 rank-1 adapters on Qwen2.5-14B; a direction from one fine-tune is used to ablate misalignment in others (https://arxiv.org/abs/2506.11618). Direction transfer is shown for activation directions via ablation, not for the adapter. | "A single rank-1 adapter suffices to induce emergent misalignment in a small LLM (Turner), and a misalignment direction extracted from one fine-tune ablates it in others (Soligo)". Also state the model sizes (0.5B, 14B). |
| 03:25 | morris2026tinylora | RL reasoning gains recoverable with tiny adapters; SFT needs much larger updates | OK | 13 parameters, 91% GSM8K; "SFT require 100-1000x larger updates" https://arxiv.org/abs/2602.04118 | none |
| 03:25 | ye2026rlvrlowrank | same sentence | OK for RL clause only | RLVR gains "primarily concentrated within the rank-1 components"; no SFT comparison https://arxiv.org/abs/2605.06523 | Put ye2026 on the RL clause only (e.g. "RLVR gains sit in rank-1 components \cite{ye}; and SFT needs 100-1000x larger updates \cite{morris}") |
| 03:26 | braga2025taskarithmetic | task arithmetic transfers domain knowledge between rerankers | overstated (minor) | Combines weights of LLMs trained on different tasks/domains by add/subtract for zero-shot adaptation of LLM re-rankers (SIGIR 2025) https://arxiv.org/abs/2505.00649 | "Task arithmetic on weights adapts LLM rerankers to new domains zero-shot" (it merges task/domain weights; it does not move a delta from one reranker to another) |
| 03:37-41 (table) | xu2024recomp, chirkova2025provence | "trained selectors and compressors cut tokens at equal quality" | overstated | "at equal quality" not checked in either abstract; Provence claims minimal drop, RECOMP claims small loss | "cut tokens with little or no quality loss" |
| 03:41 (table) | soligo2025convergent | rank-1 behaviours in LLMs | OK | see 03:24 | none |
| 02:9 | vaswani2017attention | Transformer | OK (NO) | generic | none |
| 02:9 | liu2026unifieddefinitionhallucinationits | fluent but unsupported statements analysed as failures of the world model | OK | "inaccurate (internal) world modeling" https://arxiv.org/abs/2512.21577 | none. Published title differs from arXiv; bib uses the ICML position-paper title (documented). |
| 02:11 | pmlr-v119-guu20a, 3495724.3496517 | REALM trains retriever jointly with MLM; RAG conditions generator on dense-retrieved passages | OK (NO) | generic | none |
| 02:15 | yang2018hotpotqa, trivedi2022musique | HotpotQA: 10 paragraphs, 2 supporting; MuSiQue: 2-4 hops, shortcut filtering, about 20 paragraphs | OK (NO) | matches my knowledge of the datasets | none |
| 02:15 | ho2020twowiki | 2Wiki related benchmark | OK (NO) | generic | Text says "planned for the remaining work" but chapters 04/05 already use it; update. |
| 02:19 | reimers2019sentence | bi-encoder loses interactions | OK (NO) | generic | none |
| 02:19 | nogueira2019passage | cross-encoder reads pair, typically more accurate for reranking | overstated (minor) | Nogueira shows BERT reranking beats BM25 (MS MARCO); it does not compare against bi-encoders | Cite a bi- vs cross-encoder comparison for the accuracy claim (Humeau et al., Poly-encoders, ICLR 2020; or Thakur et al., BEIR/NeurIPS D&B 2021). Not verified, suggestion only. |
| 02:19 | wang2020minilm, bajaj2016msmarco | MiniLM 6-layer, 22.7M; MS MARCO | OK (NO) | 22.7M is the standard figure for L6-H384 | none |
| 02:21 | xiong2021mdr, zhang2024beamretrieval | iterative selection follows chains at more passes | OK | see 03:16 | none |
| 02:25 | aghajanyan2021intrinsic, hu2022lora | intrinsic dimension; LoRA BA update | OK (NO) | generic | none |
| 02:32 | vasani2025phlora; ilharco2023taskarithmetic, yadav2023ties | task vectors added/subtracted "to transfer behaviour between models" | overstated | Ilharco: vectors are added to or negated from a single pretrained model; transfer is between tasks, not between models; abstract does not address cross-model transfer https://arxiv.org/abs/2212.04089 | "can be added or subtracted to add or remove behaviours of a model, or merged across fine-tunes" |
| 02:32 | (none) | Eckart-Young theorem, rank-r truncation is optimal | missing cite | PHLoRA cites Eckart & Young (1936) | add Eckart and Young, Psychometrika 1936, 1(3):211-218 (verify DOI on Springer before adding) |
| 02:45 | mcnemar1947, efron1979bootstrap, dror2018hitchhiker, card2020power | McNemar for paired binary; paired bootstrap; non-significant is not equivalence | OK (NO) | standard; Card et al. concern power, so "not evidence of equivalence" is standard inference rather than their claim | optional: cite Lakens 2017 (equivalence testing) for the non-inferiority sentence |
| 01:6 | pmlr-v119-guu20a, 3495724.3496517, liu2026... | parametric knowledge hard to update, attribute, error-prone | OK (NO) | RAG/REALM motivate exactly this | none |
| 01:6 | liu2024lostinthemiddle, wang2023filco | passages that do not support the answer can mislead the model | overstated (minor) | Lost in the Middle is about position effects and long-context degradation; FilCo supports irrelevant-context harm | add Shi et al. 2023, "Large Language Models Can Be Easily Distracted by Irrelevant Context" (ICML 2023; verify before adding) |
| 01:8 | yang2018hotpotqa, trivedi2022musique | bridge passages share little vocabulary with the question | OK (NO) | HotpotQA has bridge questions; vocabulary claim is MDR's motivation | optionally add xiong2021mdr |
| 04:8,10,15 | yang2018hotpotqa, trivedi2022musique, ho2020twowiki | datasets, 10 candidates, 2-4 supporting for 2Wiki | OK (NO) | consistent with dataset papers | none |
| 04:63 | nosek2018preregistration, card2020power | preregistration; minimum detectable effect | OK (NO) | generic | none |
| 05:9, 05:32 | lavado2027noisegate | same prompt/test questions as Noise-Gate; Adaptive-k failure on cosine | OK (internal) | see 03:10 | none |
| 05:32 | taguchi2025adaptivek | default buffer five; "without the buffer ..." | OK | see 03:10 | none |
| 06:25 | hu2022lora | LoRA ranks 1, 2, 4 | OK (NO) | generic | none |
| 06:29 | ho2020twowiki, braga2025taskarithmetic | rank-2 replicates on 2Wiki; behaviour transfer would be main novelty (cites Braga) | OK, but novelty must be re-checked | Braga is domain/task transfer for LLM rerankers | Make explicit that Braga transfers domains, not a retrieval behaviour. |

## Counts (rows above; multi-key rows counted once)

- OK (including OK(abs), OK(NO), OK(internal)): 33
- overstated: 10 (wang2023filco, zhang2024beamretrieval support recall, panigrahi, perlman, turner+soligo, braga, table row recomp/provence, nogueira, ilharco/yadav, liu2024lostinthemiddle)
- wrong: 0
- unverifiable: 0 as to content. Not re-verified on an official page: li2018intrinsic, ilharco2023taskarithmetic, hu2022lora venues (OpenReview blocked, see BIB_PUBLISHED_VERSIONS). Noise-Gate's own text was not opened; only repo notes were checked.
- Not opened at all (well-known papers, generic claims): Self-RAG, Adaptive-RAG, RankGPT, RankZephyr, MDR, Baleen, IRCoT, REALM, RAG, Vaswani, SBERT, Nogueira, MiniLM, MS MARCO, HotpotQA, MuSiQue, 2Wiki, McNemar, bootstrap, Dror, Card, Nosek.

## Venue/year consistency with bibli.bib

- shuttleworth2025illusion, sun2025dynamicrag (and liang2025draganddrop, not cited): key says 2025, `year = {2026}`, but the URL is `paper_files/2025` and NeurIPS 2025 was held in December 2025. The in-text label will read 2026. Check whether 2026 is the proceedings print year (as with Noise-Gate's 2027) and either keep it with a note or use 2025 to match the conference. I did not verify the Curran/NeurIPS page.
- lavado2027noisegate: BRACIS 2026, Springer print year 2027 (Crossref, documented). Consistent, but unusual.
- perlman2026emergence: Findings of EMNLP 2026, Budapest. Consistent with date (Nov 2026). Note: the arXiv precursor (2504.08780, "How Relevance Emerges", Nijasure, Chowdhury, Allan) has a different title and author list.
- adaptivek, beamretrieval, provence, recomp, longllmlingua, compact, vasani: venue and year match the official pages checked.

## Missing-citation spots

1. 02:32: Eckart-Young theorem has no reference.
2. 02:19: "cross-encoder is typically more accurate than bi-encoder" rests on Nogueira (BM25 comparison only).
3. 01:6 and 05:9: "irrelevant passages reduce answer quality / mislead" should cite a distraction study (Shi et al. 2023) alongside Lost in the Middle.
4. 02:9: "parametric knowledge cannot be inspected or updated piece by piece" is uncited (a knowledge-editing survey or the RAG paper would do).
5. 02:45: non-inferiority / equivalence testing is uncited (Lakens 2017, TOST).
6. 03:12: "listwise LLM rerankers are far more expensive than a small cross-encoder" cites accuracy papers only; add a cost source or state our own measured passes/latency.
7. 00_abstract: no citations, which is fine, but the sentence "Adaptive-k ..." if present should match 03/05.

## Novelty risks

Claim: "a retrieval behaviour isolated as a rank-2 change in a small reranker, with matched and random controls."

1. **Nijasure, Chowdhury, Allan, "How Relevance Emerges: A Mechanistic Analysis of LoRA Fine-Tuning in Reranking LLMs"**, arXiv 2504.08780 (extended abstract, v3 Aug 2025): https://arxiv.org/abs/2504.08780 . Same group as the cited Perlman et al. 2026. It trains LoRA ranks 1, 2, 8, 32 for reranking and ablates layers/projections. A web-search summary says rank-1 matches higher ranks, but the abstract does not say so; I could not confirm. Action: read the full paper before submission. If it shows rank-1 LoRA reaches full reranking quality, the "behaviour sits in a very low-rank update in a reranker" idea is partly pre-empted. Our distinct points would be: a 22.7M cross-encoder (not a 7B LLM), a specific behaviour (conditioning on selected evidence) rather than the whole relevance skill, extraction by SVD graft of a full fine-tune, and random/norm-matched controls. Cite it next to Perlman.
2. **Lu, Chen, Eickhoff, "Cross-Encoder Rediscovers a Semantic Variant of BM25"**, arXiv 2502.04645: https://arxiv.org/abs/2502.04645 . Mechanistic analysis of a cross-encoder (the abstract does not name the model; I could not confirm it is MiniLM). Not a pre-emption, but the closest mechanistic work on small cross-encoders. Worth citing in 03 and checking which model they use.
3. **Perlman et al. 2026 (already cited)** studies where LoRA acts in RankLLaMA and is correlational; text should say it concerns layers, not rank (see the overstated row).
4. **Soligo et al. / Turner et al.** (already cited): rank-1 adapters for a behaviour; our table row "Behaviour in tiny adapters" is accurate if the wording fix above is applied.
5. Searched for SVD-graft or functional-rank studies of retrieval behaviours in rerankers (2024-2026): none found. Related but not pre-empting: DualView (arXiv 2605.18767, https://arxiv.org/abs/2605.18767; **page says the paper was withdrawn over a dataset error, do not cite**); REPAIR, "Adaptive Retrieval for Reasoning" (ACL 2026, https://aclanthology.org/2026.acl-long.1734/), which targets bridge documents but has no weight-space analysis; TreeHop, arXiv 2504.20114 (query embedding update instead of LLM rewriting), https://arxiv.org/abs/2504.20114 (not opened, only seen in search).
6. Search coverage was shallow (three web searches). Absence of hits is not proof of novelty; consider a Semantic Scholar search on "low-rank" + "reranker" + "multi-hop".

## Response (applied 2026-11-15)

- **Adopted:**
  - FILCO sentence level; Beam Retrieval "supporting-passage F1"
  - Panigrahi (parameters only); Perlman rephrased (middle-layer attention, LLM rerankers)
  - Turner / Soligo reworded with model sizes and the activation-direction transfer
  - Ye attached to the RL clause only; Braga reworded (merging for zero-shot domain adaptation)
  - "little or no quality loss" in the positioning table
  - cross- vs bi-encoder accuracy cited to Sentence-BERT
  - task vectors reworded (add, negate, merge)
  - Eckart–Young cited (Eckart & Young 1936)
  - irrelevant-context harm cited to Shi et al. ICML 2023 (verified on PMLR), replacing Lost in the Middle
  - 2Wiki no longer "planned"
  - NeurIPS 2025 entries' year set to 2025 (proceedings URL /2025/)
- **Novelty:** Nijasure et al. (arXiv 2504.08780, read in full) train LoRA directly on 7–8B LLM rerankers for relevance; ranks 1–32, and rank 1 is often enough. They use no SVD of a full fine-tune and no conditioning or multi-hop behaviour. It is now cited in Related Work and the positioning table, with the distinction stated. Lu et al. (EMNLP 2025; BM25-like heads in a MiniLM cross-encoder) is now cited.
- **Deferred:** citations for equivalence testing and LLM-reranker cost. These are minor and stay as-is for now.
