# Final report

**Result: the submitted model is the original, unedited one** (`reset_all()`; `edited()` reports no changed tensors). I tested two edits. Neither gave a held-out gain larger than its standard error, and the stronger version made ranking worse, so under the task's rule I reverted.

## Mechanism found

- **The hard gold is the answer-bearing bridge paragraph.** In bridge questions, the lowest-scoring gold paragraph contains the answer 80% of the time. It has low question overlap (0.42) and low title match (0.34), and sits at mean rank 4.5.
- **The score is mostly lexical coverage of the question.** Within a question, the score correlates 0.60 with word overlap and 0.46 with title match. In the last layer, 7 of the 12 heads at [CLS] put 78–96% of their attention on the question tokens. So the score reads how many question words found a match, and a bridge paragraph matches only one hop of the question.
- **The final [CLS] is essentially one-dimensional.** The gold-vs-distractor direction has cosine 0.96 with the readout, and every direction I tested that is orthogonal to the readout still correlates 0.84–0.91 with the score. So editing the pooler or readout cannot help.
- **Layers 2–4 build the failure.** Along each layer's relevance axis on split B, the hard gold minus the distractors that beat it moves from +0.40 sd (after layer 2) to −0.22 (after layer 3), −0.61 (after layer 4) and −0.66 (final).
- **Layer 4's MLP is the main writer of the deficit.** Its direct attribution to that gap is −0.61 (A) and −0.70 (B), and per-neuron attributions agree across splits (corr 0.98). The responsible neurons track body overlap (corr 0.45): they are lexical-coverage detectors.

## Edits tested (split: even questions = A to choose, odd = B to confirm)

1. **Layer-5 body-reading heads h0 and h6 ×2** (`layer.5.attention.output.dense.weight`, columns 0–31 and 192–223). No effect: B tok90 −8.8 ± 14.5. The score change followed title match (0.60) and slightly hurt hard golds against beating distractors. Dropped.
2. **Layer-4 coverage neurons** (`layer.4.output.dense.weight`, 87 output columns selected on A as 60% of the negative attribution; 33,408 weights). The output columns were scaled by (1−α):
   - **α = 1 (columns zeroed):** right direction but too weak. Hard gold minus beating distractor rose on B by +0.14 (67% of questions positive). B tok90 −5.5 ± 13.9, tok85 −14.2 ± 16.9, AUC 0.641 → 0.646, both-gold@3 unchanged.
   - **α = 3 (columns × −2):** the targeted contrast grew strongly (B +0.99, 78% positive), which confirms the causal role of these neurons. But the edit acts non-linearly (10× the score change, not 3×) and degrades ranking overall: B tok90 **+57.2 ± 35.3**, AUC 0.636; A tok90 +10 ± 63.

So I located a causal mechanism, but did not find an edit within these 87 layer-4 columns that improves the metric.

## Caveats

- **Lost pass:** one of the five passes was lost when the kernel timed out mid-call, so the dose-response has only two strengths (α = 1 and 3). The useful range, if one exists, is probably α ≈ 1.5–2 or a larger neuron set at α = 1. Neither was tested, so I'm not submitting either.
- **Token metric:** token counts on practice use this model's tokenizer as a proxy for prompt tokens.

The α = 1 and α = 3 tensors, all pass scores, and the full notes are saved in `WORK` (`notes.md`, `edit_cov_a1.pt`, `edit_cov_a3.pt`).

**`edited()` at the end:** `{'track': 'R', 'heads_removed': [], 'neurons_removed': 0, 'layers_removed': [], 'scalings': 0, ...}`, with no changed tensors.