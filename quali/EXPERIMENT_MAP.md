# Experiment map for the qualification text

**Purpose:** record which experiments the text uses, and where, so the rest can be revisited later. The source of truth for each experiment is `experiments/expNN_*/README.md`. Claim status comes from `experiments/HYPOTHESES.md`.

**Story the text tells:**
1. Cheap evidence selection for multi-hop RAG.
2. Chain-following is a separable, low-rank skill in a small reranker.
3. It pays off on long chains.
4. The planned artifact is a rank-2 "chain adapter" plus a router.

## Used in the text

| Exp | What | Chapter / section | Status | Key number |
|---|---|---|---|---|
| exp02 | 22.7M cross-encoder selector (A1) on HotpotQA | Ch. 5.1 | C | 68–77% fewer tokens at equal F1 (gpt-5.4-mini) |
| exp07 | Adaptive stopping on HotpotQA | Ch. 5.2 | C | −57 / −71 tokens at recall 0.90 / 0.95 |
| exp08 | A1 on MuSiQue: the one-shot ranker is the bottleneck | Ch. 5.3 | E (primary not testable) | top-10 full recall 0.68 |
| exp09 | In-domain fine-tuning and iterative reranking | Ch. 5.3 | C?, then N at matched data | recall@5 0.682 → 0.750; confounded |
| exp10 | Adaptive stopping on MuSiQue; Adaptive-k addendum | Ch. 5.2 | C at 0.95; N for the iterative stop signal | 1,554 tokens / 20 passes at recall 0.95 |
| exp06 | A1 − A0 delta is low-rank, mid-layer | Ch. 5.4 (context) | C, mostly known | rank 4 → 80% |
| exp11 | The iteration change is rank 2 | Ch. 5.4 | C | 84.7% [74, 96] |
| exp12 | Pair-matched control; pure conditioning change is rank 2 | Ch. 5.5 | primary N, secondary C | +0.040 n.s.; rank 2 → 86.8% |
| exp13 | Fresh 4-hop confirmation; 4-hop cascade | Ch. 5.6 | C (4-hop); N (cascade) | +0.055 [+0.007, +0.102] |
| exp15 | Answer F1 on ≥ 3-hop (order-controlled) | Ch. 5.7 | C | +0.076 F1 [+0.041, +0.112]; top 5 beats all paragraphs with 69% fewer tokens |
| exp14 | Benefit router | Ch. 6 (plan) | pre-registered, waiting for GPU | — |

## Kept outside the text (for later)

| Exp | What | Why not in the text now | Revisit if |
|---|---|---|---|
| exp01 | Training-free hop gate (Noise-Gate sequel) | superseded by the learned selector | a training-free baseline is needed |
| exp03 | Linear probe on A0 | over-claimed; corrected by the PI | interpretability chapter |
| exp04 | LLM weight editing (Astra, Opus): compression ties Wanda, no repair | off the main line | a chapter on LLM-driven model editing |
| exp05 | Failure anatomy (unnamed bridge paragraphs) | motivation only; could be cited in Ch. 1 | error-analysis section |

## Pending results that will slot into the text

- **exp15:** answer F1 → Ch. 5.7.
- **exp14:** benefit router → Ch. 6.
- **Planned:**
  - rank-2 LoRA trained directly
  - transplant to another reranker
  - 2WikiMultiHopQA
  - geometry with noise baselines
