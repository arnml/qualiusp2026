# exp06 protocol: how small is the multi-hop skill? (pre-registered 2026-11-13)

## Question

A1 is A0 fully fine-tuned on HotpotQA. The fine-tuning changed almost every weight: the delta Δ = A1 − A0 touches 22.7M entries.
- **How much of A1's gain comes back** if only a small part of Δ is grafted onto A0?
- **Where does the gain live?** Embeddings, attention, MLP, which layers, or the scoring head?

**Why it matters (exp04 R1):** Astra failed to repair A0 with ≤ 108,000 changed entries in the encoder and pooler. If a graft of that size, chosen *with A1's help*, recovers much of the gain, then a small repair **exists** and Astra's limit was search. If not, the cap itself rules out a repair of that size.

## Method (no data used to choose the graft)

`graft.py`; models from `../exp04_astra_edit/models/{a0,a1}`. For every configuration, A0 + (selected part of Δ) is scored.

| Family | Configurations |
|---|---|
| **Sparse, by magnitude** (Panigrahi et al. 2023 style, without their trained mask) | top-\|Δ\| entries at 10%, 1%, 0.1%, 0.01% of all parameters |
| **Astra's scope and cap** | top-\|Δ\| 108,000 entries within `bert.encoder.layer.*` + `bert.pooler.dense.*` |
| **Random controls** | 1% random entries; 108k random entries in Astra's scope |
| **Low-rank** (PHLoRA-style truncated SVD of each 2-D delta; 1-D deltas in full) | rank 1, 2, 4, 8, 16, 32 |
| **Components, graft only** | 1-D (biases, LayerNorm), embeddings, head (pooler + classifier), all attention, all MLP, each layer L0–L5 |
| **Components, leave one out** (A1 with one group reverted to A0) | embeddings, head, each layer L0–L5 |

## Evaluation

- **Questions:** dev questions (never used to choose anything here).
  - Sweep: the first **300**.
  - Confirmation on **dev1000** for A0, A1, `astra_scope_top_108k`, `sparse_top_0.01`, and the smallest rank reaching the recovery bar.
- **Metrics:** both-gold@3, and tok@90 (exact threshold sweep on the same questions, as in exp04 Track R).
- **Recovery:** `(tok90(A0) − tok90(graft)) / (tok90(A0) − tok90(A1))`; likewise for bg3.

## Pre-registered readings (dev1000 confirmation)

1. **"A small repair exists within Astra's limits"** if `astra_scope_top_108k` recovers **≥ 50%** of A1's tok@90 gain.
   - **Under 20%:** the cap explains the R1 null.
   - **In between:** a partial answer.
2. **Sparse vs random:** magnitude grafting must beat random at the same size by more than 2 SE, or magnitude is uninformative.
3. **Low-rank:** report the smallest rank with **≥ 80%** recovery, and its parameter count.
4. **Localization:** the component tables are descriptive, with no decision rule.

## What follows

- **If reading 1 holds:** exp06 step 2. Astra gets A0, A1 (or Δ) and the same 108k cap, and must choose the graft. It is compared with magnitude, SVD and random grafts at equal size. Pre-registered separately.
- **If it does not:** R1's null is explained by the cap. Any Astra repair needs a bigger budget or a hybrid with fitting (pending the PI's decision on gradients).
