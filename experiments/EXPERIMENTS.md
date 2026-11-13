# Experiment log: cheaper multi-hop evidence selection

**Goal:** give the LLM only the evidence it needs for multi-hop QA. Use a selector that is small and cheap (few parameters and FLOPs) instead of sending all retrieved paragraphs.

**Starting point:** BRACIS 2026 Noise-Gate \cite{lavado2027noisegate}: 26.7% fewer tokens at equal F1 on HotpotQA.
**Plan and literature:** [`../research_direction.md`](../research_direction.md)

## Fixed protocol (all experiments)

- **Benchmark:** HotpotQA distractor (10 paragraphs per question: 2 gold, 8 BM25 distractors).
- **Test:** the BRACIS paper's 400 validation questions. **Dev:** 1000 other validation questions, used for tuning. Both are fixed by ID (`exp02_cross_encoder/split_ids.json`).
- **Offline metric (no LLM):** *both-gold* = both gold paragraphs kept; plus prompt tokens on the paper's exact prompt (tiktoken o200k).
- **LLM check:** stateless calls with only the selected paragraphs. Cost comes from counted prompt tokens.
- **Rules:** each experiment lives in its own folder with its own uv environment; thresholds are chosen on dev and reported once on test.

**Reference points (test):**

| Context | Tokens |
|---|---|
| All 10 paragraphs | 1442 |
| Gold paragraphs only | 323 |
| Gold sentences only | 214 |

## Status

| # | Experiment | Folder | Status | Headline |
|---|---|---|---|---|
| exp01 | Hop-aware Noise-Gate (option 1): training-free 2nd-hop threshold on "question + top paragraph" | [`exp01_hop_gate`](exp01_hop_gate/README.md) | ✅ done | 6–10% fewer tokens than Noise-Gate at equal both-gold (0.918 at 1089 tok vs 0.928 at 1221). Precision stays at ~0.25–0.30: thresholds plateau. |
| exp02 | Small learned selector (option 2): 22.7M MiniLM-L6 cross-encoder | [`exp02_cross_encoder`](exp02_cross_encoder/README.md) | ✅ done (offline + gpt-5.4-mini F1) | **F1 (gpt-5.4-mini, 400 q): A1 t=0.02 0.796 at 467 tok vs k=10 0.782 at 1448 (−68% tokens, ΔF1 +0.014, CI includes 0); top-2 0.780 at 330 tok (−77%).** Offline: | **A1 (full fine-tune): 0.915 both-gold at 391 tok (−73% vs k=10, −64% vs hop gate); 0.968 at 461 tok.** A2 n200: 0.915 at 733 tok. A0: 0.915 at 1143 tok. |
| exp03 | LLM weight editing (option 7): Opus 5.5 / Astra edit the 22M cross-encoder with N examples, compared against fine-tuning on the same N | [`exp03_weight_edit`](exp03_weight_edit/PROTOCOL.md) | ⛔ go/no-go: **no-go** for main hypothesis | Protocol pre-registered (f0f955a). Probe: the editor-allowed mean-difference direction just reproduces A0's score; even an unrestricted [CLS] probe adds only +7 pts top-3 (0.58 → 0.65), below A2 n200 (0.78). The ability is missing, not suppressed. | Go/no-go: linear probe for a "distractor direction" first. |
| exp04 | Astra (`gpt-6-astra` via Codex) edits the selector, staged B = 5 → 10 → 50 forward passes. Track C: **compress A1** (remove heads/neurons/layers) vs gradient-free pruning baselines. Track R: repair A0/A2 vs fine-tuning on the same N | [`exp04_astra_edit`](exp04_astra_edit/DESIGN.md) | ✅ **closed** ([FINDINGS](exp04_astra_edit/FINDINGS.md)). **Track C:** lossless −13% FLOPs (stage 3; dev1000 0.957 vs A1 0.959; test 0.9375 vs 0.940; CPU −9%), but a **tie with Wanda** at every stage. **Track R (A0 repair):** R1 no repair; Astra submitted no edit (its candidate failed its own confirmation; no effect on dev1000); stopped as pre-registered | Baselines (dev bg3, A1 = 0.970): Wanda-style is best: 0.970 / 0.950 / 0.930 / 0.917 / 0.820 at 90/80/70/60/50% FLOPs. Layer drop collapses (0.02 at −3 layers). Astra must beat Wanda at equal FLOPs. |
| exp05 | Failure anatomy: why A1 still misses gold paragraphs (unused validation pool, 4,000 q) | [`exp05_failure_anatomy`](exp05_failure_anatomy/README.md) | ✅ **done**. A1 bg3 0.953; all 189 misses are bridge questions; half at rank 4. The missed paragraph is the unnamed bridge entity (its title is in the question 2.6% vs 34%); a title-link feature does not separate failures from successes | Builds the A1 caches exp07 uses |
| exp06 | How small is A1's skill? Graft parts of Δ = A1 − A0 onto A0 (sparse, low-rank, per component) | [`exp06_skill_size`](exp06_skill_size/README.md) | ✅ **done**. **Low-rank, not sparse:** rank 4 (328k params) recovers 80% of the tok@90 gain; Astra's 108k-entry cap (top-\|Δ\|) recovers 3.5%, so the cap explains R1's null. Gain lives in L3–L4 | Grafts are chosen with A1 (existence, not findability) |
| exp07 | Chain-aware pair scorer + adaptive 2/3/4 paragraphs on A1's cache | [`exp07_chain_selector`](exp07_chain_selector/README.md) | ✅ **done**. **Primary succeeds:** adaptive −57 / −71 tokens at recall 0.90 / 0.95 on test (CIs below 0, recall n.s.). Secondary (pair vs top-2) n.s. (p = 0.25). F1 at 0.95: 0.789 vs 0.795 (n.s.) at −16% tokens | The gain is mostly adaptive sizing at 0.90; pair features add ~40 tokens at 0.95 (exploratory ablation) |
| exp08 | exp07 adaptive stopping on MuSiQue-Ans (2–4 hops, 20 paragraphs; A1 frozen) | [`exp08_musique`](exp08_musique/README.md) | ✅ **done**. **Pre-registered: not testable (fail)** — A1 top-10 full recall only 0.68, so k ≤ 10 cannot reach 0.90/0.95; F1 not run. Exploratory k ≤ 20: adaptive_full −62 / −100 tok vs A1 threshold (recall n.s.); title/link features add nothing; A0 worse than A1 (top-10 0.51) | One-shot reranking is the MuSiQue bottleneck (14–16 of 20 paragraphs needed); next lever is iterative retrieval |
| — | Sentence-level selection (gold sentences = 214 tok) | — | 💡 idea | — |
| — | Transfer to 2Wiki without retraining (MuSiQue: exp08) | — | 💡 idea | — |

## exp02 arms (fair comparison for exp03)

| Arm | Selector | Training data | Status |
|---|---|---|---|
| A0 | MiniLM-L6 cross-encoder, off the shelf | none (MS MARCO only) | ✅ 0.915 at 1143 tok |
| A2 | same, fine-tuned | N = 50 / 200 HotpotQA questions | ✅ n50 0.932 at 888 tok; n200 0.915 at 733 tok |
| A1 | same, fine-tuned | all ~90k train questions, 1 epoch (~50 min, T4) | ✅ 0.915 at 391 tok; 0.968 at 461; top-2 0.878 at 324 |
| A3 / A4 | same, **weights edited by an LLM** | N = 50 / 200 | exp03 |

## Findings so far

1. **Multi-hop structure matters, but cosine thresholds plateau.** The hop gate helps, yet only about 1 in 4 kept paragraphs is gold (exp01).
2. **Reading question and paragraph together helps a bit even untrained.** The zero-shot cross-encoder slightly beats the hop gate (exp02 A0).
2b. **A 22.7M fine-tuned selector closes most of the gap.** It keeps both gold in 91.5% of questions at 391 tokens (oracle 323), about 2.6 paragraphs on average. Its estimated cost is ~1/180 of the generator FLOPs it saves (exp02 A1). Small-data fine-tuning (n200) needs ~2× the tokens for the same recall, which leaves room for exp03.
2d. **Weight editing has little to work with here (exp03 probe).** A0 has no unused linear signal for multi-hop relevance that an editor could unlock; small-data fine-tuning already beats the best linear readout of A0.
2c. **Answer quality holds.** With the paper's generator, every A1 rule matches k=10 F1 (paired CIs include 0) at 68–77% fewer tokens. Gold-only context *beats* k=10 by +0.040 F1 (CI excludes 0), so distractors hurt the generator.
3. **The literature ceiling is high.** Large trained selectors pick exactly both gold paragraphs 92–97.5% of the time (SAE → Beam Retrieval). Our question is how small and cheap the selector can be.
4. **Embedder mismatch:** the BRACIS Noise-Gate numbers appear to come from OpenAI `text-embedding-3-small` (per the `cart-code` runner), not the MiniLM model stated in the paper. To be checked.

## Log

- **2026-11-09:** research direction and bibliography; exp01 run (hop gate plus 5-question Haiku pilot).
- **2026-11-10:** literature check on learned selectors; exp02 and exp03 designed.
- **2026-11-12:** exp02 pushed to GitHub and run on a Colab T4 (results saved to Drive); A0 zero-shot scored locally and matches Colab. exp04: baselines done (Wanda best); **Astra stage 1** (`gpt-6-astra` via Codex, 11 turns, 5 passes): neuron-only pruning, −23.8% FLOPs, dev300 0.963 (A1 0.970, Wanda@same FLOPs 0.950); it rejected its own head-removal and deeper-cut candidates after paired tests. A1 full fine-tune: 0.915 both-gold at 391 tok. gpt-5.4-mini F1 on the 400 questions × 5 contexts (2,000 calls, ~$0.25): A1 matches k=10 at 68–77% fewer tokens.
- **2026-11-12, night:** exp04 **stage 2** (Astra, B = 10, 15 turns, 0 violations): neuron-only, 28.57 GF (−27%), dev1000 0.940, the same as Wanda at the same FLOPs (McNemar p = 1.0). Wanda cost curve on dev1000: Wanda needs the same FLOPs as Astra for equal quality, so there is no FLOPs advantage. Astra models have 2–3% fewer params. Wanda is lossless at −15% FLOPs.
- **2026-11-12, late night:** exp04 **stage 3** (Astra, B = 50, 23 turns, 0 violations), lossless target: −13.4% FLOPs, dev1000 0.957 vs A1 0.959, test400 0.9375 vs 0.940, CPU −9%. Tie with Wanda under the pre-registered rule. Track C closed. Next (PI): Astra edits A0, the non-fine-tuned encoder (Track R).
- **2026-11-13:** exp04 **Track R stage R1** (Astra edits A0, the non-fine-tuned encoder, B = 5, practice = A2's 200 questions; pre-registered 25bc080): 16 turns, 0 violations. Astra found a layer-4 "lexical overlap" MLP candidate, failed its own predeclared confirmation (CI −18 to +49 tokens) and **submitted no edit**. Verdict: **no repair** (as the exp03 probe predicted). Exploratory dev1000 check of the rejected candidate: 1,097 vs A0 1,089 tok@90 (CI −10 to +17), no effect; the rejection was correct. A2 (same data, gradients): 698.
- **2026-11-13:** exp04 **closed and written up** in `exp04_astra_edit/FINDINGS.md`. Claims: LLM weight editing compresses (−13 to −27% FLOPs) but ties Wanda; it did not repair A0; the editor behaves like a careful experimenter (0 violations, predeclared splits, honest nulls).
- **2026-11-13, correction (PI):** exp03's "the ability is missing from A0" was an over-claim. **Correction (2026-11-13, PI):** a failed linear probe does not show the capability is absent from A0. It shows only that *this* probe (linear, [CLS], one paragraph at a time) could not extract it reliably. The information may be non-linear, distributed across layers or tokens, or recoverable only through interactions between paragraphs. Corrected in PROBE_EXPLAINED.md, PROTOCOL.md, exp04 README and FINDINGS (struck through, not deleted).

- **2026-11-13, afternoon:** **exp06 done** (README): A1 − A0 grafts. Rank 4 recovers 80.4% of the dev1000 tok@90 gain (328k params), rank 8 91.6%; top-|Δ| 108k in Astra's scope recovers 3.5% (pre-registered: under 20%, the cap explains R1's null); 1% sparse 7.6% vs random 0.3% (bg3 gap ≈ 1.8 SE, not 2). Skill mostly in L3–L4; embeddings/head/L5 ~0. **Opus 5.5 as second editor:** o1_C_B5 (Track C, 26 turns, 0 violations) −6.6% FLOPs, dev1000 0.956 vs Wanda 0.959 at the same FLOPs; o1_R_B5 (Track R, 29 turns, 0 violations) submitted no edit, replicating R1's null.
- **2026-11-13, evening:** **exp05 done:** A1 misses are hidden-bridge paragraphs, mostly at rank 4, and a title-link feature does not separate them. **exp07 done:** the pre-registered primary succeeds (adaptive −57/−71 tok at recall 0.90/0.95 on test), the secondary is n.s. F1 follow-up with gpt-5.4-mini (1,600 calls, $0.13): ΔF1 −0.020 [−0.045, +0.005] at 0.90 and −0.006 [−0.027, +0.015] at 0.95. All A1-based conditions sit at 0.77–0.80 F1, so the practical gain over a fixed top-2 is recall/reliability, not average F1. exp07 bug fixed: `np.load` NpzFile re-decompressed the cache per question.
- **2026-11-13, night:** **exp08 done** (MuSiQue-Ans, pre-registered 0d1aa07). Pre-registered primary/secondary **not testable**: A1's top-10 holds all supporting paragraphs for only 0.68 of dev questions (A0: 0.51 on test), so the k ≤ 10 policy cannot reach the targets; the F1 follow-up was not run. Exploratory k ≤ 20: adaptive_full −62 [−97, −27] / −100 [−131, −68] tokens vs the A1 threshold at recall 0.90 / 0.95 (recall n.s.); `text` = `a1only`, so title/link features add nothing and the gain is from A1's vectors. Reaching 0.90–0.95 needs 14–16 of ~20 paragraphs (1,700–2,000 of 2,535 tokens).
