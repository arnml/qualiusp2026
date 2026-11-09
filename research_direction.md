# Research direction: a cheaper multi-hop selector / compressor

Draft, 2026-11-09. Continues the BRACIS 2026 paper \cite{lavado2027noisegate} (code and full text: `C:\code\cart-code`).

**Goal (one sentence):** build something small, such as a network, an embedding adapter or a selector, that decides which retrieved paragraphs the LLM sees in multi-hop QA. It should keep answer quality while costing far less than sending all candidates to the LLM, measured in tokens, FLOPs and parameters.

It does not need to "understand" or "reason". It only needs to be cheap and good enough at choosing evidence.

## 1. Starting point: what Noise-Gate already gives us

Setup in the paper: gpt-5.4-mini as generator, `multi-qa-MiniLM-L6-cos-v1` embeddings (22M parameters, 384-d, cached), HotpotQA distractor with n=400, plus 2Wiki and MuSiQue with n=100 each.

| HotpotQA | F1 | Tokens |
|---|---|---|
| No retrieval | 0.387 | 104 |
| Fixed k=3 | 0.651 | 471 |
| Fixed k=10 (reference) | 0.792 | 1453 |
| **Noise-Gate τ=0.30** | **0.764** | **1065 (−26.7%)** |

**Headroom.** One paragraph costs about 135 tokens ((1453 − 104) / 10). HotpotQA has only **2 gold paragraphs out of 10**, so a gold-only context would be about 375 tokens. Noise-Gate still sends about 7 paragraphs, so roughly 5 of them are noise. If gold-only context keeps quality, the ceiling is **about −70% tokens**, against −27% today.

→ **First experiment (cheap, 1 day):** an oracle run with gold paragraphs only, using the same generator and samples. It measures the real ceiling and frames every result that follows.

**Why one threshold stops at −27%:**
1. The second-hop (bridge) paragraph is often *not* similar to the question; it is similar to the first-hop paragraph. To keep it, τ has to be low, and a low τ lets distractors in.
2. Paper §6: BM25 distractors on the same topic sit in the same cosine range as gold paragraphs, so no single number separates them.

What is missing is (a) a **hop-aware** signal and (b) a **nonlinear** decision. Both can still be very cheap.

## 2. Options, ordered from smallest step to largest

The selector cost column assumes the paragraph embeddings are already cached, as they are now.

| # | Option | What is new | Extra params | Selector cost / query | Training | Novelty | Risk |
|---|---|---|---|---|---|---|---|
| 1 | **Hop-aware Noise-Gate** | After keeping the top paragraph p1, run a second threshold on `cos(p, q + p1)` (bridge expansion, as in MDR \cite{xiong2021mdr}), only over cached vectors | 0 | ~10 extra dot products | none (tune τ1, τ2) | low–medium | very low |
| 2 | **Tiny learned gate (MLP)** | A nonlinear keep/drop over features we already have: `e_q, e_p, e_q⊙e_p, cos(q,p), max cos(S,p), max Jaccard(S,p), BM25 rank, dense rank`, with greedy set selection | 10^4–10^5 | ~1 MFLOP | HotpotQA `supporting_facts` labels (free) | medium | low |
| 3 | **Embedding adapter** ("new embedding") | A small linear or low-rank map on the query embedding (and optionally on `q + p1`) so that gold paragraphs clear τ. The Noise-Gate algorithm stays unchanged | 384×r (~10^4) | 1 matrix-vector product | same free labels, contrastive loss | medium | low |
| 4 | Small cross-encoder / pruner | Reads (q, p) jointly, as Provence \cite{chirkova2025provence} does | 22M–140M | ~10× encoder forward passes | yes | low (exists) | low |
| 5 | Sentence-level compressor | Keeps sentences, not paragraphs (RECOMP \cite{xu2024recomp}, BRIEF \cite{li2025brief}) | 10^8+ | high | yes | low (exists) | medium |
| 6 | Soft-token compression | One embedding per document fed straight into the LLM (xRAG \cite{cheng2024xrag}) | projector | very low | yes | high | high: needs an **open-weights** generator, not gpt-5.4-mini |
| 7 | **LLM-written weights** | A frontier LLM writes or edits the weights of option 2 or 3 directly \cite{takehi2026weightspace} | same as 2/3 | same as 2/3 | 0 to ~50 forward passes | high | high, see §5 |

Options 4 and 5 are **baselines**, not contributions: they already exist and are 1000× larger than option 2. Option 6 is a good PhD direction once there is an open generator.

## 3. Recommended small step for the qualification

**Main contribution: options 1, 2 and 3 as one family of selectors that work on cached embeddings and are hop-aware.**

- Option 1 is the training-free, hop-aware Noise-Gate. It is the direct sequel to BRACIS and can be implemented in an afternoon in `cart-code`.
- Option 2 is the learned version. Training labels are HotpotQA's own supporting-fact annotations, so **no LLM cost to make the gate**. Training takes minutes on CPU.
- Option 3 is the same idea expressed as a better embedding instead of a classifier. It is useful if the gate has to be swapped into another retriever.

Optional label variant: replace the gold labels with "useful-to-the-LLM" labels (leave-one-out with the generator, as ECoRAG \cite{jeong2025ecorag} and the utility selector \cite{zhang2025utilityselector} do). This costs LLM calls once, at training time only.

**Exploratory arm (option 7):** ask Opus 5.5 / Astra to write the gate weights of option 2 or 3 directly. Compare against the trained gate on the same features. It is cheap to try, and the result is interesting either way, as long as it follows the audit protocol of \cite{ahmed2026weightspaceaudit}.

**Target claim:** *a gate with fewer than 10^5 parameters on cached embeddings stays within the ±0.05 F1 band of fixed k=10 with ≥50% fewer tokens (Noise-Gate: 26.7%), and its selector cost per query is about 10^6× smaller than letting an LLM choose.*

## 4. Evaluation: fair cost accounting

Keep the BRACIS protocol: matched samples, fixed generator, ±0.05 F1 band around k=10.

- **Quality:** F1 (main), EM, supporting-paragraph recall and precision (the gate's own accuracy, cheap to compute, no LLM needed).
- **Generator cost:** input tokens and USD on gpt-5.4-mini. Its FLOPs are unknown because it is an API model, so:
- **Add one open-weights generator** (for example a ~8B model) to report generator FLOPs ≈ 2 × N_params × tokens. This also answers the paper's stated limitation of a single generator.
- **Selector cost:** parameters and FLOPs per query, plus the one-time cost to build the gate (training FLOPs, or LLM tokens for options 7 and the label variant), amortized per query.
- **Main figure:** F1 vs. total cost per query (Pareto). Points: k = 3/5/10, Noise-Gate, options 1–3, Provence, LLM-as-selector (upper bound), and the gold oracle.

Datasets: HotpotQA distractor (main), 2WikiMultiHopQA and MuSiQue (transfer). Ideally at full validation size, since the BRACIS pilots were n=100.

## 5. Where weight-space programming fits, and its risk

- Claim \cite{takehi2026weightspace}: frontier LLMs read and write network weights without gradient descent. Successes are on **small** networks; edits remove failure modes rather than add skills. MIT CSAIL blog post, not peer reviewed.
- Counterpoint \cite{ahmed2026weightspaceaudit}, arXiv:2611.08070 (2026-11-05): tests **Claude Opus 5.5** reading ~125k-parameter CNNs. The reading looks like local arithmetic plus memorized dataset priors, not tracing. It provides an audit protocol.
- For us this matters less than it seems. We don't need the LLM to *understand* the gate, only to produce weights that work. Options 2 and 3 are small enough (10^4–10^5 parameters) to sit in the range where the claimed successes are. If it fails, the trained gate is still the contribution.
- Trained alternatives that generate weights: Text-to-LoRA \cite{charakorn2025texttolora} and Drag-and-Drop \cite{liang2025draganddrop}. These are PhD material, for example a hypernetwork that emits a domain-specific gate from a short description of the domain.

## 6. Related work map (all in `bibli.bib`)

- **Our baseline:** Noise-Gate \cite{lavado2027noisegate}; Adaptive-k \cite{taguchi2025adaptivek}; set selection \cite{lee2025setr}.
- **Multi-hop retrievers (hop-aware idea):** MDR \cite{xiong2021mdr}, Beam Retrieval \cite{zhang2024beamretrieval}.
- **Small or distilled selectors:** ECoRAG \cite{jeong2025ecorag}, utility selector 32B→1.7B \cite{zhang2025utilityselector}, Context-Picker \cite{zhu2025contextpicker}, DynaKRAG \cite{zhou2026dynakrag}.
- **Compressors / pruners:** Provence \cite{chirkova2025provence}, RECOMP \cite{xu2024recomp}, BRIEF \cite{li2025brief}, LongLLMLingua \cite{jiang2024longllmlingua}, xRAG \cite{cheng2024xrag}.
- **Weight-space:** \cite{takehi2026weightspace}, \cite{ahmed2026weightspaceaudit}, \cite{charakorn2025texttolora}, \cite{liang2025draganddrop}.
- **Cost control (agent side):** TokenEconomizer \cite{li2026tokeneconomizer}.

## 7. Next steps

1. Oracle gold-only run on HotpotQA (n=400), to measure the ceiling.
2. Option 1 (hop-aware Noise-Gate) in `cart-code`, with a τ2 sweep.
3. Option 2 (MLP gate) trained on HotpotQA train `supporting_facts`; evaluate gate recall and precision first (free), then F1 with the generator.
4. Option 3 (embedding adapter) as a variant of option 2.
5. Option 7 pilot: the LLM writes the option-2 weights; compare at equal features.
