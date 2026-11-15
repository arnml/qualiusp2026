# Next directions: one-page decision (2026-11-13)

The full reasoning, literature and review of the second LLM's roadmap are in git history (`74a0194`). This page is the condensed version.

## Where we stand

| What | Result |
|---|---|
| exp02 trained selector A1 | **68–77% fewer LLM tokens at the same answer F1.** Our main practical result |
| exp04 Astra compresses A1 | 13–27% cheaper at near-equal quality, but **ties Wanda** pruning |
| exp04 Astra repairs A0 | **no repair** in one 5-pass run |
| Evidence check (exp02 data) | if a gold paragraph is missing, F1 drops 0.78 → 0.48, so **evidence matters** |
| Correction (PI) | the exp03 probe does **not** show A0 lacks the skill; only a linear [CLS] readout failed |

**The limits that shape the next steps:**
- **Pointwise scoring:** our selector scores each paragraph alone, so it can't see that two paragraphs complete a chain.
- **The repair rules were tight:** Astra had no fitting, a 1% edit cap and 5 passes, and the edit was MLP-only by its own choice.

## Candidates, ranked (research value first, practical value second)

| # | Experiment | Question | Research | Practical | Cost |
|---|---|---|---|---|---|
| **1** | **exp06 How small is the skill?** | Copy the largest 1% / 0.1% / 0.01% of A1's changes, or a rank-1–16 SVD of them, into A0. How much of A1 comes back? Then: does Astra choose *which* parts better than SVD or random? | ★★★ where a multi-hop skill lives; a fair, informative task for Astra | ★ | CPU hours; Astra later |
| **2** | **exp07 Chain-aware adaptive selector** | A tiny pair scorer on A1's cached features judges whether the top paragraphs *complete each other*, and decides to send 2, 3 or 4 | ★★ chain awareness with no extra cross-encoder passes | ★★★ fewer tokens per question | CPU + Colab |
| 3 | exp05 Failure anatomy | Why does A1 still miss gold paragraphs? Feeds 1 and 2 | ★ (enabler) | ★ | CPU, an afternoon |
| 4 | Opus 5.5 as a second editor | Does another editor behave like Astra? | ★★ removes the one-editor limit | – | ~1 h CPU + Claude usage |
| 5 | Interaction surgery (MICE-style) | Cut query–paragraph interactions for real latency | ★★ | ★★ | Colab, largest |

## Recommendation

- **Now:** run **exp06 step 1** (no LLM, CPU) and **exp05** (CPU) together. Both are cheap.
  - exp06 answers: "is there a small fix for Astra to find?"
  - exp05 answers: "what is A1 still getting wrong?"
- **Then, depending on the results:**
  - **If a small graft recovers most of A1:** give Astra the weight delta and compare it with SVD and random allocation. That's a strong research result either way.
  - **In parallel:** start exp07 for the practical, multi-hop story, tested on HotpotQA and on **MuSiQue**, where pointwise selection fails more.
- **For the thesis:** exp02 (practical) + exp04 (an honest weight-editing study) + exp06 or exp07 (the new contribution).

## Rules carried over

- **Pre-register** each decision rule in a commit before running.
- **Paired tests and non-inferiority margins** for "equal".
- **A large unused validation pool** when the headroom is small: A1 misses only ~40 of 1,000 dev questions.

## Open question for the PI

**Are gradients allowed from now on?**
- **Claude's view:** yes. The thesis constraint is cheap *inference*, not gradient-free development, and exp04 already tested the pure, gradient-free version.
- **What it enables:** Astra chooses where and what to edit, and a tiny gradient fit sets the values (LoFiT-style). This is compared with LoFiT and LoRA at equal parameter counts.

## Update (2026-11-13, evening): candidates 1–4 are done

| # | Result | Consequence |
|---|---|---|
| 1 exp06 | **Low-rank, not sparse:** rank 4 recovers 80%; a 108k sparse cap recovers 3.5%; the skill sits in L3–L4 | Astra's R1 null is explained by the edit *shape*. "Small fix in A0" exists only as a low-rank, dense change, which saves no inference |
| 2 exp07 | Adaptive 2–4 paragraphs: **−16% tokens at recall 0.95**, F1 unchanged (n.s.) | A practical, cheap add-on to A1. Pair features help mostly at high recall |
| 3 exp05 | Misses are unnamed bridge paragraphs, half at rank 4 | Explains where exp07 helps; title links alone are not enough |
| 4 Opus | Same picture as Astra: ties Wanda, no A0 repair, 0 violations | The exp04 findings are not specific to one editor |

**Suggested next steps (for the PI to choose):**
- **a. exp07 on MuSiQue / 2Wiki:** the real test of chain awareness, and the practical thesis line.
- **b. Astra repair v2:** a rank-≤ 4 budget in L3–L4. Needs the gradients decision.
- **c. Interaction surgery** (candidate 5) for real latency.

## Update (2026-11-13, night): exp08 = exp07 on MuSiQue

- **Pre-registered outcome: not testable.** A1's top 10 holds all supporting paragraphs for only 68% of MuSiQue questions (A0: 51%). The k ≤ 10 policy chosen from HotpotQA cannot reach recall 0.90/0.95.
- **Exploratory (k ≤ 20):** the adaptive MLP still saves 62–100 tokens vs the A1 threshold at equal recall (n.s.). Title/link features add nothing; the signal is A1's vectors.
- **Consequence:** on 2–4-hop questions the bottleneck is the **one-shot ranker**, not the stopping rule. Sending 0.95 recall still costs ~2,000 of 2,535 tokens.

**Suggested next steps:**
- **a. Iterative / query-updated reranking:** re-score the remaining paragraphs with the question + selected paragraphs (A1 or a small fine-tune on MuSiQue train). This is the direct fix for the 4-hop recall collapse, and the natural PhD line.
- **b. Fine-tune A1 on MuSiQue train** (a cheap upper bound on what one-shot ranking can do there), then rerun exp08 unchanged with k ≤ 10.
- **c. Astra repair v2** and **interaction surgery** as before.

## Update (2026-11-14): exp09 iterative reranking done

- **Iteration works when trained:** recall@5 0.750 vs 0.682 one-shot on the same data (p = 0.0067), and 4-hop 0.58 vs 0.36. At recall 0.90 it needs 1,425 tokens vs 1,533 (one-shot) vs 1,957 (A1).
- **The bigger lever is in-domain training** (A1 → M1: −424 tokens at 0.90). Iteration costs 3.7× the passes and loses at 0.95 (greedy 4 steps; weak tail).

**Suggested next steps:**
- **a. Combine exp07 + exp09:** adaptive stopping on top of the iterative ranker. The stop decision is natural after each greedy step ("is the chain complete?"); it should cut both tokens and passes.
- **b. Fix the tail:** merge one-shot and iterative scores, or beam 2; and a pair-matched M1 control (the threat noted in exp09).
- **c. F1 with gpt-5.4-mini** on M1-iter vs M1 vs A1 at recall 0.90 (≈ 1,200 calls, < $0.20).
- **d. 2WikiMultiHopQA** as a third dataset for the thesis.

## Update (2026-11-15): exp10 done; exp11 pre-registered

- **exp10:** adaptive stopping saves −244 tokens over fixed k at recall 0.95 on M1-iter, but **one-shot M1 + adaptive is cheaper still** (1,554 tok, 20 passes). The iterative "anything left?" score is not a better stop signal. Practical line: **in-domain one-shot ranker + adaptive stop**.
- **Research line chosen (user: "explore research value"):** **exp11**: is the change that turns M1 into M1-iter (the iteration skill) low-rank and in L3–L4, like the HotpotQA skill (exp06)? The domain skill (A1 → M1) is the contrast.
- **Still open:** the tail fix (merge one-shot + iterative scores), the F1 check, 2Wiki.

## Update (2026-11-15, afternoon): exp11 + exp12 done

- **Practical line:** in-domain one-shot ranker + adaptive stop. Iteration buys only +0.03 recall@5 at matched data (n.s.) for 3.7× the passes.
- **Research line, sharpened:** the *ability to use a conditioned query* is a separable rank-2 change (86.8% recovery, de-confounded). Its value appears on 4-hop chains (+0.19, exploratory).
- **Candidate next experiments (ranked):**
  1. **Cascade / switchable rank-2 adapter:** one-shot by default, the rank-2 conditioning adapter only for long or hard questions. Pre-register the 4-hop effect as a confirmatory test on a new question set (MuSiQue train holdout), plus tokens, passes and F1 vs Adaptive-k.
  2. **Geometry with noise baselines (no training):** spectra, stable rank, principal angles across the 3 seed deltas (exp12) and D_it / D_dom / A1−A0; α-scaling.
  3. **Transplant:** graft the rank-2 conditioning delta onto other one-shot rerankers (A1-HotpotQA, ms-marco MiniLM, the seed-1/2 ctrl models).
  4. **Train LoRA rank 1–2 directly** (Colab) vs post-hoc SVD: functional vs trained rank (TinyLoRA contrast).
