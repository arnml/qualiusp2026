# Synthesis of three independent reviews (2026-11-15)

**Sources:**
- [`../LITERATURE_POSITIONING.md`](../LITERATURE_POSITIONING.md): literature agent, about 45 citations. Claude checked four key ones on arXiv: 2602.04118, 2608.23338, 2505.00649, 2506.08479.
- [`AI_ASSISTED_RESEARCH_PRACTICES.md`](AI_ASSISTED_RESEARCH_PRACTICES.md): methodology agent.
- [`codex_review.md`](codex_review.md): Codex (OpenAI, high reasoning effort, read-only), acting as a critical colleague. It was **not blind** to our hypotheses.

## 1. Where the reviews agree

| Point | Literature | Codex | Practices |
|---|---|---|---|
| exp06 (low-rank, mid layers) is **mostly known**: Aghajanyan, LoRA, PHLoRA, Perlman et al. 2026 | ✅ | ✅ | – |
| Adaptive stopping (exp07/08/10) is **known**: Adaptive-k, DynamicRAG, CompAct. We must compare against the Adaptive-k gap rule | ✅ | – | – |
| The exp09 method is **known** (Beam Retrieval, NAACL 2024). New: it works at 22M params with 1k questions | ✅ | ✅ | – |
| **exp11 is the most novel result**: a *behavior* (iteration) as a rank-2 delta between two rerankers, with a random control. But it is **descriptive and confounded** | ✅ | ✅ | – |
| **Main confound:** D_it mixes iteration with "more training". M1-iter saw 2.2× the pairs. A **pair-count-matched one-shot control** is needed | ✅ | ✅ (its #1) | – |
| **One seed, one dataset:** a replication gate (seeds and a second dataset) is needed before a thesis claim | ✅ | ✅ | ✅ |
| Practical claims need **answer F1** at a token budget, not only recall | – | ✅ | – |
| **"CI includes 0" is not equivalence.** Use non-inferiority margins | – | ✅ | ✅ |

## 2. What this means for the thesis contribution

- **Research claim (candidate):** in a small cross-encoder, query-conditioned multi-hop iteration is a separable, low-rank, transferable change, unlike domain relevance, which is localized.
  - This is new only if it survives the matched control and seeds.
  - Prior work that bounds it: TinyLoRA shows reasoning in rank ≤ 1 under **RL**, and says supervised fine-tuning needs 100–1000× larger updates. Our supervised iteration delta is rank 2, which is an interesting contrast.
- **Practical claim:** a tiny iteration adapter, switched on only when needed, gives iterative recall on hard questions at close to one-shot cost. It must be measured in LLM tokens, cross-encoder passes and answer F1, against Adaptive-k.

## 3. Next experiments (ranked; each pre-registered separately)

| # | Experiment | Why first | Cost |
|---|---|---|---|
| **1** | **exp12 matched control:** train a one-shot model from A1 with the same pair count and steps as M1-iter (M1-matched, 18,101 pairs). Then test: (a) M1-iter vs M1-matched, does iteration still win? (b) is M1-iter − M1-matched still rank ≈ 2 and different from M1-matched − M1? | Without it, exp09's primary and exp11's claim stay confounded. All three reviewers flagged this | About 5 h CPU (one seed) |
| 2 | **Free geometry (no training):** singular-value spectra and stable rank, principal angles between D_it, D_dom and A1−A0 (against a random baseline), an α-scaling curve | Sharpens "rank 2" into a statement about the geometry of the change; it costs minutes, except α (about 25 min per point) | 1–3 h |
| 3 | **Skill transplant:** graft the rank-2 iteration delta onto other one-shot rerankers (A1 on HotpotQA, ms-marco-MiniLM-L-6-v2) | The most novel and practical test: is the skill portable? Closest prior: Braga et al. SIGIR 2025 (domain, not behavior) | Hours, no training |
| 4 | **Cascade:** one-shot, plus the rank-2 adapter only when needed. Compare tokens, passes and F1 against Adaptive-k and fixed k | The practical payoff of #3 | Hours, plus about $0.2 for F1 |
| 5 | 2Wiki replication of the final claim | Replication gate | Hours |

## 4. Process changes

Adopted now (cheap; they apply from exp12 on):
1. **`HYPOTHESES.md` claim ledger:** every test is labelled confirmatory or exploratory, with its status. Confirmatory tests are the pre-registered primaries only. Exploratory wins must be re-tested before they are claimed.
2. **A power / minimum-detectable-effect line** in every PROTOCOL.md (Card et al. 2020).
3. **Blind adversarial review:** Codex reviews each PROTOCOL.md *before* the run without being told the hypothesis we hope for, and argues for the null.
4. **Citation rule:** every cited paper is checked to resolve (arXiv, ACL or DOI) before it enters the thesis.
5. **`AI_USE.md`:** a log of which tools did what (venue disclosure policies).
6. **A non-inferiority margin** stated in advance for every "equal" claim.

Proposed, needing the PI or user:
- **Locked confirmation set:** a reserved question set opened only once, by the human, for the final thesis claims.
- **PI sign-off on each PROTOCOL.md before it runs.**
- **A replication gate:** two or more seeds and a second dataset for every headline claim.
