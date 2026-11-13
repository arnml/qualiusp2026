# Final edit: 2 heads and 840 neurons removed, cost 39.26 → 36.66 GFLOPs per question (−6.6%)

The edit keeps quality on practice questions: the selection half nets zero, and the confirmation half is unchanged. It is modest: two bolder versions each cost a question, and dropping layer 1 failed badly.

**Practice split:** fixed random permutation (seed 0); 100 questions to choose removals (A), 100 to confirm (B). All rankings below were computed from the original model's outputs; nothing was fitted or trained.

## What I removed

**1. 700 layer-5 MLP neurons.**
- **Why it's safe:** in the last layer only the [CLS] position reaches the pooler. So a neuron's whole effect is its output at [CLS] times its output-weight column, which I can read directly from its recorded activations and weights.
- **How chosen:** I ranked neurons by RMS output at [CLS] on split A times the norm of their output column, and removed the 700 smallest. The removed vector is about 6–7% of the residual scale entering the final LayerNorm.
- **Lesson learned:** my first ranking used only within-question variance. It picked neurons sitting at the GELU floor (~−0.16), whose near-constant outputs added up to an offset of ~44% of that scale. That caused real damage in pass 2. Ranking by RMS includes the constant part and avoids this.

**2. Layer-5 attention heads 11 and 5.**
- Only their [CLS] row matters in this layer.
- They have the smallest direct write onto the decision direction: 0.05 and 0.08, against up to 0.30 for other layer-5 heads. I measured this as the OV circuit projected onto a gold-vs-distractor class-mean direction (from split A), passed through the LayerNorm gains.

**3. 140 neurons in layers 0–3** (24 / 92 / 14 / 10 per layer).
- Each has |output| < 0.005 at [CLS] on every one of the 1,992 practice items, so they never fire at [CLS].
- **Caveat:** this criterion used activations from all items, including split B.
- Measured effect: [CLS] cosine to the original stays ≥ 0.9991 through layer 4.

## Effect on practice questions (pass 3)

| | Original | Edited | Changes |
|---|---|---|---|
| Split A (selection) | 0.91 | 0.91 | 1 lost, 1 gained |
| Split B (confirmation) | 0.98 | 0.98 | none |

- Within-question Spearman against the original is 0.987 (A) and 0.988 (B).
- Mean |score change| is 0.21, against a score std of 5.
- Within-question score-difference std is 0.25. A Gaussian noise model on the original scores puts this at about −0.3 pp expected change.

## What I tried and rejected

- **Dropping layer 1** (pass 2, combined with the bad layer-5 ranking): A fell 7 points, B fell 17. Layer 1 had the smallest [CLS] update of any layer, yet dropping it damaged the token representations. At layer 4's output the gold-vs-distractor separation fell from 0.70 to 0.51. So I did not try dropping any other layer.
- **Pass 4** (pass 3 plus 1,000 layer-5 neurons, layer-5 head 8, L4h9, L1h6, L1h7, and early neurons with max |output| < 0.02): would save 34.69 GFLOPs. Net −1 question on A, no change on B, but score noise doubled to 0.55. Perturbations entering at layer 1 grew 2.5–4× per layer.
- **Pass 5** (pass 3 plus 1,000 layer-5 neurons, layer-5 head 8, L4h9): would save 35.47 GFLOPs. Noise was 0.40, B unchanged, but A still netted −1 (question 13, the same loss as pass 4). I had decided beforehand to adopt it only with no net loss, so I reverted to pass 3.

No attention head was a clean [SEP]/punctuation sink (at most 0.48 of its attention), so that route gave few candidates.

Final `cost()`: **36.66 GFLOPs per question, 9,902,328 encoder parameters** (2 heads and 840 neurons removed, no layers, no rescalings). My notes are in the session's `notes.md` in the work folder.