# Where we are and what to try next (2026-11-13)

Partly-correct reasoning by design: these are hypotheses to test, not conclusions. Short version first, details below.

## In one minute

- **What worked:** a trained selector (A1, exp02) cuts LLM tokens by 68–77% at the same answer F1. Astra then made it 13–27% cheaper (FLOPs) at near-equal quality, but simple Wanda pruning does the same.
- **What didn't:** Astra could not improve A0, the untrained encoder, in one 5-pass run.
- **Correction (PI):** the exp03 probe does **not** show A0 lacks the skill. It shows only that a linear [CLS] readout can't find it. The signal may be non-linear, spread over layers or tokens, or need **two paragraphs at once**.
- **The biggest structural limit:** our selector scores each paragraph **alone**. A bridge paragraph is often only relevant *given* the other gold paragraph. No weight edit can fix that in a model that never sees both.
- **Recommended next steps,** all built on your three questions:
  1. adaptive compute (cheap, practical)
  2. "does a small fix exist?" (cheap, decides whether more Astra repair runs are worth it)
  3. a pair-aware selector (the real multi-hop question)

## What the edits actually touched

| Run | What Astra tried | What it kept |
|---|---|---|
| Track C, stages 1–3 (compress A1) | read attention weights and 8 attention maps per run; tried removing heads (8, then 2) and whole layers (1, 2, 5); rescaled columns | **only MLP neurons removed**; heads and layers always hurt, so it rolled them back |
| Track R, R1 (repair A0) | looked at 8 attention maps; edited 96 columns of the **layer-4 MLP output** (36,864 weights) | **nothing**: the edit failed its own confirmation test |

- **Never edited:** attention weights (Q/K/V/O) were never changed in Track R.
- **Why the prompt nudged it towards MLPs:**
  - the "no fitting" rule rules out computing new attention patterns
  - the 1% cap (108k weights) is small
  - 5 passes allow about one hypothesis and one confirmation
- **Can we change this? Yes.** Prompt variants can require an attention-level hypothesis, allow only attention edits, raise the cap, or allow closed-form updates (next section).

## Why A0 repair may have failed (hypotheses, each testable)

| Hypothesis | What would show it | Cost |
|---|---|---|
| **H1. A small fix exists, but Astra couldn't find it** under 5 passes / no fitting | "Grafting": copy the 1% (or 0.1%) largest weight changes from A1 into A0. Panigrahi et al. (ICML 2023) found ~0.01% of fine-tuned weights recover >95% of fine-tuning gains on their tasks | CPU, ~1 h, no LLM |
| **H2. The signal is in A0 but not linear at [CLS]** (your correction) | Non-linear probe (small MLP), all layers, token-pooled features, 1,000 questions | CPU, ~1 h |
| **H3. The task needs two paragraphs at once** (bridge reasoning) | A pair model `[q; p_i; p_j]` beats pointwise, especially on bridge questions and harder datasets (MuSiQue) | Colab training |
| **H4. Finding where to edit ≠ knowing how to edit** | Hase et al. (NeurIPS 2023): localization results did not predict the best layer to edit | literature; affects how we design Astra's role |

H1 and H2 are cheap and decide whether more Astra repair runs are worth it:
- **If 1% grafting recovers most of A1:** a small repair exists, and Astra's limit was search. Then Astra-with-more-freedom is promising.
- **If grafting fails at 1%:** the cap itself is the obstacle, and more Astra runs under the same rules are unlikely to work.

## Your three questions → three experiments

### 1. Adaptive retrieval: easy questions cost almost nothing (exp05, recommended first)

- **Idea:** a cascade.
  - A very cheap scorer runs first, such as the pruned A1, or A1 exiting at layer 2–3.
  - If it is confident (a clear gap between the 2nd and 3rd paragraph scores), stop and send 2 paragraphs.
  - If not, run the full model and send more paragraphs.
- **Connects to:** the BRACIS Noise-Gate (adaptive k), exp01's hop gate, and exp04's finding that the last MLP is redundant.
- **Measure:** average FLOPs and tokens per question vs answer F1, compared with the fixed A1.
- **Expected:** most HotpotQA questions are easy for A1 (0.96 both-gold@3), so the average cost should drop a lot. Honest risk: the confidence signal may be weakest exactly on bridge questions.
- **Practical value: high.** Cost is proportional to difficulty, on CPU and in LLM tokens.

### 2. "Does a small fix exist?", then Astra repair v2 (exp06)

1. **Step 1, no LLM:** grafting tests H1 at 1% / 0.1% / 0.01% of weights, and a non-linear probe tests H2. Both are cheap and decide what comes next.
2. **Step 2, only if a small fix exists:** a fairer Astra repair run, with three changes:
   - **Astra as the localizer, a small fit for the values.** Astra picks which heads or neurons to change, and a tiny fit on the 200 questions sets the values. This is the LoFiT recipe (Yin et al. 2024: tuning offsets on 3–10% of heads matches LoRA with 20–200× fewer parameters).
     - *Question:* does an LLM pick better places than LoFiT's own selection or random places?
     - *Honest prior (H4):* placement often matters less than people think.
   - **More freedom:** B = 50, cap 1–10%, attention edits explicitly encouraged.
   - **A "transplant" variant:** give Astra the A1 checkpoint too, and ask it to find and explain the 1% of A1's changes that matter. Compare with magnitude grafting.

### 3. A small model that understands relationships between paragraphs (exp07)

- **Pointwise (today):** `score(q, p)`. The bridge paragraph looks irrelevant on its own.
- **Pair-aware options, cheapest first:**
  - **Two-hop rescoring:** take the top-1 paragraph, append its title to the question, and rescore the others. No new model; MDR and Beam Retrieval do this with trained models (Xiong et al. 2021; Zhang et al. 2024).
  - **Pair cross-encoder:** input `[q; p_i; p_j]`, used only on the top-4 candidates (6 pairs, so ~0.6× A1's full cost on top of a cheap first stage).
  - **Set/listwise selector** over all paragraphs (SetR, Lee et al. 2025).
- **Where it should help:** HotpotQA distractor leaves little room for A1 (0.96 both-gold@3; 385 tokens vs 323 for an oracle). The real test is **MuSiQue and 2WikiMultiHopQA**, where pointwise selection fails more often.
- **Link to Astra:** once the input contains both paragraphs, "small targeted edits that improve relationship understanding" become possible in principle. Today they aren't.

## Should we try the Opus family as editor?

- **For:**
  - It removes our biggest limitation: one editor and one run per stage. "Careful experimenter" is a claim about LLM editors in general only if a second model behaves the same way.
  - The harness needs one new class (`claude -p` with tools off; the same prompt, kernel and audit).
- **What to run:** replicate Track C stage 1 (B = 5) and R1 (B = 5) with Opus 5.5. Small, comparable, and about 1 h of CPU.
- **Disclosure:** I am Opus 5.5, so my opinion here may be biased. The run decides, not me.
- **Expected honestly:** similar compression results (Wanda-level). On A0 repair, the same constraints probably give the same null; it becomes interesting after exp06 step 1.

## Value, honestly

| | Research value | Practical value |
|---|---|---|
| exp02 trained selector | solid, known technique, clean evidence on cost vs F1 | **high:** 68–77% fewer LLM tokens |
| exp04 Astra compression | **useful negative/neutral result:** first pre-registered test of LLM weight editing on a retrieval model *against strong pruning baselines*; one mechanism finding (last MLP redundant, heads essential) | low: Wanda gives the same in seconds; the selector is ~1/180 of the LLM cost it saves |
| exp04 Astra repair (R1) | weak alone (one run, 5 passes); valuable once H1/H2 are tested | none yet |
| exp05 adaptive cascade | moderate (adaptive compute for RAG is active) | **high** |
| exp06 grafting + Astra v2 | **high if positive** (an LLM finds a sparse skill edit); a clean negative is also publishable | uncertain |
| exp07 pair-aware selector | **high for the multi-hop thesis** | high on hard datasets |

**Context for exp04's value:** the weight-space programming results we follow are recent, unreviewed, and partly disputed by an audit (Ahmed 2026). A careful, baseline-matched replication on a new model type is a real contribution even when the answer is "ties classical pruning".

## Suggested order (smallest step, biggest information first)

1. **exp06 step 1:** grafting + non-linear probe. CPU only, a few hours. Decides the Astra-repair question.
2. **exp05:** adaptive cascade. CPU plus existing models; strong practical story for the qualification.
3. **Opus replication** of stage 1 and R1, if we want the editor claim to generalise.
4. **exp07:** pair-aware selector on MuSiQue / 2Wiki (Colab).
5. **Astra repair v2:** only if step 1 shows a small fix exists.

## Sources

- Panigrahi et al., *Task-Specific Skill Localization in Fine-tuned Language Models*, ICML 2023: <https://arxiv.org/abs/2302.06600>
- Yin, Ye & Durrett, *LoFiT: Localized Fine-tuning on LLM Representations*, NeurIPS 2024: <https://arxiv.org/abs/2406.01563>
- Hase et al., *Does Localization Inform Editing?*, NeurIPS 2023: <https://arxiv.org/abs/2301.04213>
- Wu et al., *ReFT: Representation Finetuning for Language Models*, NeurIPS 2024: <https://arxiv.org/abs/2404.03592>
- Weight-space programming (Takehi et al. 2026, unreviewed) and its audit: `bibli.bib` (takehi2026weightspace, ahmed2026weightspaceaudit); press summary <https://analyticsindiamag.com/ai-news/mit-researchers-show-llms-can-read-and-edit-neural-network-weights-without-retraining>
- Already in `bibli.bib`: MDR (xiong2021mdr), Beam Retrieval (zhang2024beamretrieval), SetR (lee2025setr), adaptive-k (taguchi2025adaptivek), Noise-Gate (lavado2027noisegate).
