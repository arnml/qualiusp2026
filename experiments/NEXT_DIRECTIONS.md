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
