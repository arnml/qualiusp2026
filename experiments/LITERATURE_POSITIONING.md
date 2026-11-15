# Literature positioning: small cross-encoder evidence selection, iteration, and low-rank skill deltas

Compiled 2026-11-15. Every entry below was checked against arXiv / ACL Anthology / proceedings pages, except
those marked (mem). Entries marked (mem) are well-known papers whose IDs I am confident of but did not re-open
in this pass. "Rel." = why it matters to us.

## 1. Positioning (one paragraph)

Our work sits where three literatures meet. (a) **RAG context selection and compression** (RECOMP, LongLLMLingua,
FILCO, CompAct, Provence, Adaptive-k, DynamicRAG, DPS) aims to send fewer tokens to the LLM. These methods mostly use
LLM-sized or DeBERTa-sized components, report compression ratio or answer EM/F1, and rarely report "LLM tokens at fixed
evidence recall" with a ~20M-parameter CPU model. (b) **Multi-hop retrieval** (MDR, Baleen, Beam Retrieval, IRCoT,
EfficientRAG, GRITHopper, contextual passage utility) already conditions on previously selected passages. Our exp09
iterative reranker is **a known idea** (Beam Retrieval feeds question + chosen passages + candidate to a cross-encoder),
here reproduced at 22.7M params with a cost lens. (c) **Structure of fine-tuning deltas** (intrinsic dimension, LoRA, task
vectors, delta compression, skill localization, rank-1 RL updates) shows that deltas are often low-rank and partly
localized. What we found no published precedent for is **isolating one IR behavior (query-conditioning on
already-picked evidence) as the delta between two fine-tuned rerankers**, showing it is rank about 2 with a matched-norm
random control and spread across layers, unlike the base relevance skill, and then **using that delta as a
transferable, switchable adapter**. The defensible thesis contribution is: *"Multi-hop iteration is a rank-2, portable
skill in small rerankers; it lets you get iterative-reranker recall and LLM-token savings at near one-shot training
and serving cost."* The transfer/portability part is not yet shown and is the top recommended experiment.

## 2. Per-question tables

### Q1. Low-rank / intrinsic-dimension structure of fine-tuning deltas; skill localization

| Paper | Link | Rel. (one line) |
|---|---|---|
| Li, Farkhoor, Liu, Yosinski. *Measuring the Intrinsic Dimension of Objective Landscapes*. ICLR 2018 (mem) | [arXiv:1804.08838](https://arxiv.org/abs/1804.08838) | Origin of "intrinsic dimension" via random subspace training. |
| Aghajanyan, Zettlemoyer, Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ACL 2021 (mem) | [arXiv:2012.13255](https://arxiv.org/abs/2012.13255) | Fine-tuning works in a few hundred random dims, so low-dim deltas are expected. Uses random subspaces, not SVD of the actual delta. |
| Hu et al. *LoRA: Low-Rank Adaptation of LLMs*. ICLR 2022 (mem) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Rank 1–8 adapters match full FT. Includes an analysis showing ΔW amplifies directions not emphasized in W. |
| Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023 (mem) | [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) | Task vectors (θ_ft−θ_pre) can be added, negated, and composed. This is the template for "skill as a delta". |
| Yadav et al. *TIES-Merging*. NeurIPS 2023 (mem) | [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) | Merges task vectors by trimming and sign-resolving. Baseline for composing deltas. |
| Panigrahi, Saunshi, Zhao, Arora. *Task-Specific Skill Localization in Fine-tuned LMs*. ICML 2023 | [arXiv:2302.06600](https://arxiv.org/abs/2302.06600) | Grafting about 0.01% of weights recovers most FT performance (**sparse** localization). Our exp06 contradicts this for our setup: sparse top-magnitude recovers only 3.5%, low-rank recovers 80%. Note that they *optimize* the mask, while we used magnitude, so this is a confound to flag. |
| Liu et al. *BitDelta: Your Fine-Tune May Only Be Worth One Bit*. NeurIPS 2024 | [arXiv:2402.10193](https://arxiv.org/abs/2402.10193) | Deltas compress to 1 bit per weight. Deltas are highly compressible. |
| Ping et al. *Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision*. NeurIPS 2024 | [arXiv:2406.08903](https://arxiv.org/abs/2406.08903) | Long-tailed delta singular values, so pure low-rank loses accuracy on LLM math/code. Contrast: our deltas are *very* low-rank. |
| Vaish et al. *PHLoRA: data-free post-hoc LoRA extraction from full-rank checkpoints*. Findings IJCNLP-AACL 2025 | [arXiv:2509.10971](https://arxiv.org/abs/2509.10971) | Truncated SVD of the delta gives a servable LoRA (rank 32 is about full). Exactly our exp06/exp11 method, used for serving, so the *method* is not novel. |
| Biderman et al. *LoRA Learns Less and Forgets Less*. TMLR 2024 (mem) | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) | Full-FT deltas on code/math have rank 10–100× higher than typical LoRA. Low rank is task-dependent, not universal. |
| Shuttleworth, Andreas, Torralba, Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence*. 2024 | [arXiv:2410.21228](https://arxiv.org/abs/2410.21228) | Spectral analysis: LoRA introduces "intruder dimensions" and full FT does not. Relevant if we retrain the skill as LoRA (Exp B). |
| Mukherjee, Yuan, Hakkani-Tür, Peng. *RL Finetunes Small Subnetworks in LLMs*. NeurIPS 2025 | [arXiv:2505.11711](https://arxiv.org/abs/2505.11711) | RL updates are sparse (5–30% of params) but nearly full-rank. Sparse vs low-rank depends on the regime. |
| Morris, Mireshghallah, Ibrahim, Mahloujifar. *Learning to Reason in 13 Parameters* (TinyLoRA). 2026 | [arXiv:2602.04118](https://arxiv.org/abs/2602.04118) | RL reasoning gains with rank-1 or smaller adapters. So "a reasoning behavior is rank 1–2" is **already known for LLMs under RL**. |
| Ye et al. *Implicit Reward Overfitting and Low-rank Dynamics in RLVR*. 2026 | [arXiv:2605.06523](https://arxiv.org/abs/2605.06523) | RLVR gains are concentrated in the rank-1 component of updates. Same caveat as above. |
| Lu, Chen, Eickhoff. *Pathway to Relevance: How Cross-Encoders Implement a Semantic Variant of BM25*. EMNLP 2025 | [arXiv:2502.04645](https://arxiv.org/abs/2502.04645) | Mechanistic circuit in an MS MARCO cross-encoder: TF/IDF-like signals in early-to-middle layers, IDF in a low-rank embedding direction. Supports our exp06 "L3–L4" localization. |
| Perlman, Nijasure, Allan. *Emergence of Relevance Through Axiomatic Attention Patterns During LoRA Fine-Tuning*. Findings EMNLP 2026 | [arXiv:2608.23338](https://arxiv.org/abs/2608.23338) | RankLLaMA: LoRA restricted to a compact mid-network region recovers more than half the gain. **Closest prior to exp06** (layer localization of reranker FT), but it uses a 7B model, layer-restricted LoRA rather than delta grafting, and studies relevance, not iteration. |
| Braga, Kasela, Raganato, Pasi. *Investigating Task Arithmetic for Zero-Shot IR*. SIGIR 2025 | [arXiv:2505.00649](https://arxiv.org/abs/2505.00649) | Adds a domain delta to IR-tuned rerankers (T5, Llama-2), giving up to +18% NDCG@10. **Closest prior to "skill transplant across rerankers"**, but it transplants *domain*, not a *behavior*. |
| Cross-LoRA (2025); LoRA-X (2025) | [arXiv:2508.05232](https://arxiv.org/abs/2508.05232), [arXiv:2501.16559](https://arxiv.org/abs/2501.16559) | Training-free LoRA transfer across base models via subspace alignment. Relevant if we transfer across different checkpoints. |

**Answer to the sub-questions.** (i) A specific behavior living in a rank-1/2 delta has been shown for LLM *RL reasoning*
(TinyLoRA, RLVR rank-1), not, as far as I found, for a *retrieval/reranking behavior in a small encoder*. (ii) We found no
paper composing or swapping a *behavior* adapter (multi-hop conditioning) across rerankers. Braga et al. 2025 swap
*domain* vectors, and Cross-LoRA swaps generic adapters across LLMs. I am fairly but not fully confident this gap is
real. A citation-graph check of Braga et al. and Perlman et al. is recommended before claiming it.

### Q2. Multi-hop / iterative retrieval and reranking

| Paper | Link | Mechanism | Model / cost reported | Rel. |
|---|---|---|---|---|
| Asai et al. *Learning to Retrieve Reasoning Paths* (PathRetriever). ICLR 2020 (mem) | [arXiv:1911.10470](https://arxiv.org/abs/1911.10470) | Recurrent graph walk over Wikipedia links, BERT reranking of paths | BERT-base, beam over paths | Early "condition on previous hop". |
| Xiong et al. *Multi-Hop Dense Retrieval* (MDR). ICLR 2021 (mem) | [arXiv:2009.12756](https://arxiv.org/abs/2009.12756) | Query updated as q + passage₁, then dense retrieval for hop 2 | RoBERTa bi-encoder, beam 2 hops; reports 10× faster than graph methods | Query-update idea, bi-encoder. 2-hop only. |
| Khattab, Potts, Zaharia. *Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval*. NeurIPS 2021 (mem) | [arXiv:2101.00436](https://arxiv.org/abs/2101.00436) | Condenser keeps only selected facts per hop. Queries are q + condensed facts | ColBERT + ELECTRA condenser | "Condensed retrieval" is the conceptual twin of our `q [SEP] picked`. |
| Zhang et al. *End-to-End Beam Retrieval for Multi-Hop QA*. NAACL 2024 | [arXiv:2308.08973](https://arxiv.org/abs/2308.08973), [ACL](https://aclanthology.org/2024.naacl-long.96) | **Cross-encoder over question + selected passages + candidate**, beam search, variable hops, end-to-end | DeBERTa-v3 large/base; SOTA support recall on MuSiQue/HotpotQA/2Wiki; cost scales as beam × candidates × hops | **Direct prior of exp09.** Must cite. Our version is 22.7M, greedy, plus a token lens. |
| Trivedi et al. *Interleaving Retrieval with CoT* (IRCoT). ACL 2023 (mem) | [arXiv:2212.10509](https://arxiv.org/abs/2212.10509) | LLM CoT step drives each retrieval | Many LLM calls per question | Expensive baseline that our approach avoids. |
| Zhuang et al. *EfficientRAG*. EMNLP 2024 | [arXiv:2408.04259](https://arxiv.org/abs/2408.04259) | Small Labeler + Filter generate next-hop query without LLM calls | DeBERTa-sized modules; fewer LLM calls and tokens on HotpotQA/MuSiQue/2Wiki | **Closest "cheap iterative" competitor**. Compare tokens and recall. |
| Erker et al. *GRITHopper: Decomposition-Free Multi-Hop Dense Retrieval*. 2025 | [arXiv:2503.07519](https://arxiv.org/abs/2503.07519) | Generative-embedding multi-hop dense retriever, with Beam Retriever as cross-encoder reference | 7B embedder | Recent SOTA context; notes cross-encoder iteration cost. |
| Jain, Garimella. *Modeling Contextual Passage Utility for Multihop QA*. IJCNLP-AACL 2025 | [arXiv:2512.06464](https://arxiv.org/abs/2512.06464) | Small transformer scores passage utility *in context of other passages*, trained from reasoning-model traces | "small transformer" (size not on abstract page) | **Second direct prior** for context-conditioned small reranking. Must cite and compare. |
| Meng et al. *From Ranking to Selection: Dynamic Passage Selector* (DPS). 2025 (preprint) | [arXiv:2508.09497](https://arxiv.org/abs/2508.09497) | Set-level selection modeling inter-passage dependencies, variable set size | Fine-tuned LLM; MuSiQue F1 +30% over Qwen3-reranker | LLM-sized set selector. Ours is 22M. |
| Schlatt et al. *Set-Encoder: Permutation-Invariant Inter-Passage Attention*. ECIR 2025 | [arXiv:2404.06912](https://arxiv.org/abs/2404.06912) | Cross-encoder with inter-passage [INT] tokens | Electra-base scale, one pass | One-pass alternative to our sequential iteration. Possible cheaper competitor. |

**Takeaway.** Conditioning a cross-encoder on already-picked passages is established (Beam Retrieval, Baleen, contextual
utility). What is less covered: a **22M CPU model**, explicit **cross-encoder passes vs LLM tokens** accounting, and
**the 4-hop gain** (0.36 to 0.58) at that size.

### Q3. Context selection / compression / adaptive-k / stopping

| Paper | Link | Rel. |
|---|---|---|
| Liu et al. *Lost in the Middle*. TACL 2024 (mem) | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | Long or noisy contexts hurt answers, which motivates sending fewer passages. |
| Xu, Shi, Choi. *RECOMP*. ICLR 2024 (mem) | [arXiv:2310.04408](https://arxiv.org/abs/2310.04408) | Extractive and abstractive compressors. Metric: compression rate + EM. Includes a selective-augmentation (send nothing) option. |
| Jiang et al. *LongLLMLingua*. ACL 2024 (mem) | [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) | Token-level prompt compression with a small LM plus document reordering. Standard token-reduction baseline. |
| Wang et al. *Learning to Filter Context for RAG* (FILCO). 2023 (mem) | [arXiv:2311.08377](https://arxiv.org/abs/2311.08377) | Trained context filter at sentence level, evaluated on HotpotQA and others. |
| Asai et al. *Self-RAG*. ICLR 2024 (mem) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | LLM decides when to retrieve and critiques passages. Expensive adaptive baseline. |
| Jeong et al. *Adaptive-RAG*. NAACL 2024 (mem) | [arXiv:2403.14403](https://arxiv.org/abs/2403.14403) | Small classifier routes by query complexity (no / one / multi-step retrieval). Analogue of our stop and route decisions. |
| Yoon et al. *CompAct: Compressing Retrieved Documents Actively*. EMNLP 2024 | [arXiv:2407.09014](https://arxiv.org/abs/2407.09014) | Segment-wise compression with **early termination**. Evaluated on HotpotQA/MuSiQue/2Wiki. Competitor for "stop when enough". |
| Chirkova et al. *Provence: Efficient and Robust Context Pruning for RAG*. ICLR 2025 | [arXiv:2501.16214](https://arxiv.org/abs/2501.16214) | DeBERTa **reranker + pruning head** in one model at near-zero overhead. **Closest "cross-encoder as token saver"** prior. Sentence-level, not multi-hop iterative. |
| Taguchi, Maekawa, Bhutani. *Efficient Context Selection for Long-Context QA: ... Adaptive-k*. 2025 | [arXiv:2506.08479](https://arxiv.org/abs/2506.08479) | Largest-gap cutoff over similarity scores, training-free, up to 10× fewer tokens. **Must-beat baseline for exp07/08/10.** |
| Sun, Zhong, Zhou, Han. *DynamicRAG*. 2025 | [arXiv:2505.07233](https://arxiv.org/abs/2505.07233) | RL-trained LLM reranker with a STOP action, giving variable k with LLM-feedback reward. Learned-stop competitor at 7B scale. |
| Sun et al. *Is ChatGPT Good at Search?* (RankGPT). EMNLP 2023 (mem) | [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) | Listwise LLM reranking with sliding windows. High-cost reference. |
| Pradeep, Sharifymoghaddam, Lin. *RankZephyr*. 2023 (mem) | [arXiv:2312.02724](https://arxiv.org/abs/2312.02724) | Open 7B listwise reranker. |
| Yoon et al. *ListT5*. ACL 2024 (mem) | [arXiv:2402.15838](https://arxiv.org/abs/2402.15838) | FiD-based listwise reranker. Mid-size listwise reference. |

**Standard metrics/baselines.** Retrieval: recall@k, *full-support recall* (all gold paragraphs in the set; MuSiQue/2Wiki),
HotpotQA supporting-fact EM/F1. End task: answer EM/F1 with a fixed reader. Efficiency: compression ratio (RECOMP,
CompAct, LongLLMLingua), input tokens, latency, number of LLM calls (EfficientRAG, DynamicRAG). **"LLM tokens at fixed
recall/F1"** (our metric) is uncommon but easy to defend. Report it alongside compression ratio and EM/F1 so readers can
compare. Standard baselines: fixed top-k (k=3,5,10), Adaptive-k gap rule, BM25/dense + off-the-shelf cross-encoder
(e.g. ms-marco-MiniLM, bge-reranker), and one LLM compressor (RECOMP or LongLLMLingua).

## 3. What is novel vs known in our results (honest)

| Our result | Status | Notes |
|---|---|---|
| A1: small CE selector cuts LLM tokens 68–77% at equal F1 (HotpotQA) | **Known in spirit** | Provence, RECOMP, Adaptive-k, and FILCO show token cuts at equal quality. Our value is the 22.7M CPU scale and the clean tokens-at-equal-F1 accounting. Engineering contribution, not a research claim. |
| exp06: HotpotQA delta low-rank (rank-4 recovers 80%), magnitude-sparse fails, mid layers L3–L4 | **Mostly known** | Low-rank deltas: Aghajanyan, LoRA, PHLoRA. Mid-layer reranker skill: Perlman et al. 2026, Lu et al. 2025. The "low-rank ≫ sparse" contrast with Panigrahi is interesting but confounded, since they *optimize* the mask. Supporting result only. |
| exp07/08/10: learned adaptive-k beats fixed k | **Known** (Adaptive-k, DynamicRAG, CompAct early stop, Adaptive-RAG) | Must be compared against the Adaptive-k gap rule. The negative finding (iterative "anything left?" score is not a better stop signal) is a small but honest new observation. |
| exp09: iterative `q [SEP] picked` reranking, +6.8 recall@5, 4-hop 0.36 to 0.58 | **Method known** (Beam Retrieval, Baleen, contextual utility) | New: works at 22.7M with 1k training questions, and zero-shot iteration collapses. Report the 3.7× CE-pass cost explicitly. |
| exp11: one-shot to iterative delta is **rank 2 (95%)**, random rank-4 at matched norm gives 9%, spread across layers (unlike exp06) | **Likely novel in IR / small encoders** | Low rank for a *reasoning behavior* is known for LLM RL (TinyLoRA, RLVR rank-1), so "rank 2" alone is not surprising. Novel: (a) the behavior is defined as the delta between two *fine-tuned rerankers*; (b) the matched-norm random control; (c) the localized (relevance) vs distributed (iteration) contrast inside one model. This is still descriptive. It becomes a contribution once the delta is shown to be **portable and useful** (Section 4). |

**Threats to validity to pre-empt.** (1) Rank-2 recovery measured only on MuSiQue dev. Needs held-out test and 2Wiki.
(2) The one-shot and iterative models share initialization and data, so a small delta may reflect a short fine-tune
rather than the skill's intrinsic simplicity. Control: delta of a same-length *non-iterative* continued fine-tune.
(3) The magnitude-sparse baseline is weak. Add Panigrahi-style optimized grafting for a fair comparison.

## 4. Ranked next experiments (CPU-feasible, MiniLM-L6, MuSiQue/HotpotQA/2Wiki)

| Rank | Experiment | Novelty × practical | Must cite / beat |
|---|---|---|---|
| **1** | **Skill transplant: graft the rank-2 iteration delta onto other one-shot rerankers.** Targets: (a) HotpotQA A1, (b) a 2Wiki-trained one-shot model, (c) off-the-shelf `ms-marco-MiniLM-L-6-v2`. Add α·ΔW_r2 (tune α on dev) and evaluate iterative recall@5, 4-hop recall, and LLM tokens at recall 0.90/0.95. Controls: zero-shot iteration (known to collapse), random rank-2 at matched norm, full iterative FT on the target (upper bound), and a TIES/task-arithmetic full-delta graft. Optional: Cross-LoRA-style subspace alignment when the targets diverge. | **High × high.** If it works, "iteration" ships as a few-KB adapter that turns any MiniLM reranker iterative without multi-hop training data. No IR paper transplants a *behavior* delta. | Braga et al. 2025 (task arithmetic for IR), Ilharco 2023, TIES 2023, PHLoRA 2025, Cross-LoRA 2025. Beam Retrieval as the iterative reference. |
| **2** | **Train iteration directly as rank-2 LoRA, with a data-efficiency curve** (100/250/500/1000 MuSiQue questions; full FT vs LoRA r=1,2,4). Measure iteration gain, *forgetting* of one-shot ranking (one-shot recall with adapter off is trivially intact, so measure one-shot recall *with adapter on*), and the intruder-dimension spectrum. | **Medium × high.** Gives a cheap recipe (train only about 0.1% of params on CPU) and tests whether "rank 2 post-hoc" equals "rank 2 learnable". | Hu 2022, Biderman 2024 (learns less, forgets less), Shuttleworth 2024 (intruder dims), TinyLoRA 2026. |
| **3** | **Compute-aware cascade with a switchable adapter.** One model, two modes: one-shot pass first. Apply the rank-2 iteration adapter (merged on the fly, roughly free) only when the one-shot score profile is uncertain (gap rule / exp10 stop model) or the question is predicted multi-hop (Adaptive-RAG-style router). Report the Pareto frontier of **LLM tokens vs CE passes vs full-support recall vs answer F1** on MuSiQue + 2Wiki + HotpotQA. | **Medium × very high.** Directly addresses the 3.7× CE-pass cost of exp09 and gives the thesis its practical headline. | Adaptive-k 2025 (gap rule), Adaptive-RAG 2024, EfficientRAG 2024, CompAct 2024, Provence 2025, contextual utility 2025, Beam Retrieval 2024, Set-Encoder 2025 (one-pass alternative). |

**Suggested order.** Run Exp 1 first, since it is cheap (no training beyond what exists, just grafting and evaluation) and
it decides the narrative. If transplant works, Exp 3 becomes "portable iteration adapter + cascade" (strong thesis). If it
fails, Exp 2 + 3 still give a solid "rank-2 LoRA iterative selector at one-shot cost" contribution, with exp11 as the
mechanistic motivation.

**Must-have baselines across all three.** Fixed top-k. The Adaptive-k gap rule. An off-the-shelf MiniLM / bge-reranker.
Beam Retrieval numbers (cite their reported MuSiQue/HotpotQA/2Wiki recall; do not re-run the large model). One LLM
compressor reference point (RECOMP or LongLLMLingua) via reported numbers.
