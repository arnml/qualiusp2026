# exp04 findings: can a frontier LLM edit a retrieval selector's weights?

**Status: closed (2026-11-13).** Track C was run to completion (stages 1–3). Track R stopped after R1, as pre-registered.
- Details and every number: [README.md](README.md)
- Design, amendments and pre-registrations: [DESIGN.md](DESIGN.md)

## Question

Our multi-hop selector is a 22.7M-parameter cross-encoder (MiniLM, 6 layers). It chooses which of 10 HotpotQA paragraphs an LLM reads. Can a frontier model, `gpt-6-astra` via Codex, improve it **by reading and editing its weights directly**, with no gradients and only a handful of forward passes?

1. **Track C, compress:** make the fine-tuned selector A1 cheaper (fewer FLOPs and parameters) at the same quality.
2. **Track R, repair:** make the off-the-shelf encoder A0 good at the task.

## Method in one paragraph

We followed the harness of Takehi et al. (2026, Exp 8) and kept its seven-section prompt.
- **One code block per turn:** each Astra turn may contain one Python block, run in a persistent CPU kernel with a small API (`load`, `remove_*`, `save_edit`, `cost`, `forward`).
- **Budgeted forward passes:** B = 5 / 10 / 50.
- **Isolation and audit:** all built-in Codex tools are off and audited per turn. Code that uses gradients, fitting or networking is refused before it runs.
- **Practice data:** 200 questions the editor may use. Every result below is on held-out questions the editor never saw: dev1000, and test400 once at the end.
- **Pre-registration:** each stage's success criterion was committed before the run (commits `a967db1`, `8a39213`, `25bc080`).
- **Baselines:** pruning methods given the same 200 questions (Wanda-style |W|·‖x‖, Taylor, random, layer drop), and for repair, fine-tuning on the same 200 questions (A2).

## Results

### Track C: compress the fine-tuned selector A1 (A1 = 0.959 dev1000 both-gold@3, 39.26 GFLOPs)

| Stage | Passes | Astra's edit | GFLOPs | dev1000 | Wanda at equal FLOPs | Wanda's FLOPs for equal quality | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 3,584 MLP neurons | 29.90 (−24%) | 0.946 | 0.945 (McNemar p = 1.0) | ~29.8 | tie |
| 2 | 10 | 4,096 MLP neurons | 28.57 (−27%) | 0.940 | 0.940 (p = 1.0) | 28.56 | tie |
| 3 | 50 | 2,016 neurons, incl. the **whole layer-5 MLP** | 34.00 (−13%) | **0.957** | 0.962 (p = 0.40) | ~32.8 | tie (pre-registered margin 2 GF) |

**Stage 3 on test400 and CPU** (removed parts physically deleted; exact to 1e-5):

| | Params | test400 | CPU ms per question |
|---|---|---|---|
| A1 | 10.65M | 0.940 | 241 |
| Astra | 9.10M | 0.9375 | 219 |
| Wanda | 9.17M | 0.9425 | 217 |

**Findings:**
1. **Cheaper at the same quality: yes.** Astra's stage 3 edit is lossless within noise (−0.2 points dev, −0.25 test), with −13% FLOPs, −15% params and −9% CPU time.
2. **Better than classical pruning: no.** Wanda-style pruning ties it at every stage, and more passes (5 → 50) did not open a gap. Taylor and random pruning are clearly worse (−4 to −8 points at equal FLOPs).
3. **Qualitatively different solutions:**
   - Astra never removed attention heads (Wanda always does).
   - Astra found that the last layer's MLP can be removed entirely while that layer's attention cannot. This mechanism finding was confirmed by its own ablations.
   - Its edits have 1–3% fewer parameters than Wanda's at the same FLOPs.
4. **Whole layers are not removable:** dropping one layer costs ~20 points.

### Track R: repair the off-the-shelf encoder A0 (stage R1, 5 passes)

| Model | dev1000 both-gold@3 | tokens to keep both gold for 90% |
|---|---|---|
| A0 (start) | 0.593 | 1,089 |
| **Astra R1 submitted edit** | = A0 (no edit submitted) | = A0 |
| Astra's rejected candidate (exploratory) | 0.594 | 1,097 (CI −10 to +17) |
| A2: A0 fine-tuned on the same 200 questions | 0.769 | 698 |

**Findings:**
- **No repair**, as the exp03 linear probe predicted. Astra found a plausible mechanism (layer-4 MLP neurons that favour lexically overlapping distractors), but silencing them did not survive its own predeclared confirmation, so it submitted no edit.
- On 1,000 held-out questions, the rejected candidate has no effect. Its discovery-half gain was overfitting to 100 questions.
- **Track R stopped at the pre-registered R1 gate.**

### Second editor: Opus 5.5 replication (pre-registered 2026-11-13, added after closure)

Identical harness, prompts, budgets and evaluation; `claude -p` with all tools off (see DESIGN.md).

| | Astra stage 1 (`s1_C_B5`) | **Opus `o1_C_B5`** |
|---|---|---|
| Turns / passes / violations | 11 / 5 / 0 | 26 / 5 / 0 |
| Edit | 3,584 MLP neurons | 700 L5 MLP neurons + L5 heads 5, 11 + 140 never-active early neurons |
| GFLOPs | 29.9 (−24%) | **36.66 (−6.6%)** |
| dev1000 both-gold@3 | — | **0.956** (A1 0.959) |
| Wanda / Taylor / random at the same FLOPs | — | 0.959 / 0.959 / 0.948 |
| McNemar vs Wanda | — | 1 vs 4 discordant, **p = 0.375** (tie) |

**Pre-registered reading (cost at equal quality, 2 GF margin):**
- Wanda reaches ≥ 0.956 at 33.22 GF (85% FLOPs, 0.959); the conservative curve point 34.35 GF (87.5%, 0.956) is also cheaper.
- Opus needs 2.3–3.4 GF more, so it **loses** to Wanda on cost.

**Track R (`o1_R_B5`, 29 turns, 5 passes, 0 violations, 1 soft-flag turn):** **no repair.**
- Opus found layer-4 MLP output columns whose silencing raises the hard gold above the beating distractor (B: +0.14 at α = 1; +0.99 at α = 3, so causal).
- Neither strength improved the metric (B tok@90 −5.5 ± 13.9 at α = 1; +57 at α = 3).
- One pass was lost to a kernel timeout. It submitted **no edit**, so the model is A0 by construction (same category as R1; `evaluate_r.py` not needed for the verdict).

**What replicates across editors:**
- 0 tool violations
- predeclared rules and rollbacks
- the last-layer MLP being removable
- a null A0 repair with the right mechanism-level diagnosis

**What differs:** risk appetite. Opus made a 4× smaller cut and rejected −12% candidates that lost one practice question each.

**Why the repair fails (from exp06):** A1's skill is low-rank and spread over L3–L4. The 108k largest entries of A1 − A0 recover only 3.5% of the gain, so a sparse ≤ 108k-entry edit is the wrong shape.

## What we can claim

1. **LLM weight editing can compress, but not beat pruning.** With 5–50 forward passes, a frontier model produces a selector that is 13–27% cheaper at (near-)equal quality. A simple gradient-free baseline (Wanda) does the same.
2. **LLM weight editing did not repair A0, in this setting.** With 5 passes, a 1% edit cap and no fitting allowed, it found no edit that improves the off-the-shelf encoder. This does not show the ability is absent from A0 (see the exp03 correction); only that this editor, under these constraints, did not find a repair. Fine-tuning on the same 200 questions saves ~390 tokens per question.
3. **The editor is a careful experimenter.** Over 4 sessions and 65 turns:
   - 0 tool violations and 0 refused code
   - predeclared discovery/confirmation splits, paired tests and rollbacks of edits that failed confirmation
   - honest null results (R1)
   - honest misses reported (stage 2's 77/80 vs 78/80)
4. **Interpretable by-products:** heads matter more than their attention maps suggest. The final MLP is redundant for ranking. The A0 failure is not a single linear direction or MLP subset.

## Limitations

- **Two editors, one run per stage:** Astra and Opus 5.5 (stage-1 budget only for Opus). There is no seed variance, and the run-to-run spread of each editor is unknown. The analysis is written by the same model family as the second editor.
- **Sample size:** dev1000 has ±0.7 points of noise. All "ties" are within it, and the Wanda curve is not monotone above 80% FLOPs.
- **Isolation:** Windows has no OS sandbox. Rule compliance rests on the code audit and the logs (all committed under `runs/`).
- **Track R was stopped at 5 passes:** more passes or a larger edit cap might behave differently. The probe prior and the null candidate make that unlikely, but it is untested.
- **Compute:** Codex usage was about 12.4M input / 0.55M output tokens over the four runs, about 7% of the weekly ChatGPT-plan limit for Track C. Wall-clock time was dominated by CPU forward passes.

## Run index

| Run | Track | B | Turns | Codex input tokens | Editor time |
|---|---|---|---|---|---|
| `s1_C_B5` | C | 5 | 11 | 1.20M | 8 min |
| `s2_C_B10` | C | 10 | 15 | 2.50M | 9 min |
| `s3_C_B50` | C | 50 | 23 | 5.85M | 14 min |
| `r1_R_B5` | R | 5 | 16 | 2.78M | 9 min |
| `o1_C_B5` (Opus) | C | 5 | 26 | 1.33M (Claude) | 17 min |
| `o1_R_B5` (Opus) | R | 5 | 29 | 1.89M (Claude) | 23 min |

## References

Takehi et al. 2026; Ahmed 2026; Sun et al. 2024 (Wanda); Michel et al. 2019; Voita et al. 2019; Sajjad et al. 2023; Gromov et al. 2024; Alain & Bengio 2016; Hewitt & Liang 2019 (keys in `bibli.bib`).
