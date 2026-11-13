Pass 2 (body heads ×2) had only a marginal effect, within noise:
- A: AUC 0.720 → 0.724, tok85 830 → 809, tok90 flat.
- B: AUC 0.641 → 0.646, tok85 912 → 885, tok90 flat.
- Both-gold@3 changes are +2/−0 (A) and +2/−2 (B).

So the body-reading heads have little leverage on [CLS]; the score is set elsewhere. (The traceback was a format-string bug in my last print, not a model problem.)

Before spending another pass I'll look inside the coverage heads. For the 8 recorded inputs (question 0), which question tokens does each layer-5 head (and layer-4 h6/h7/h9) read from [CLS]? I'm looking for whether some read entity names (what distractors share) and others the relation/answer-type words (what bridge golds share).

```python
# Hypothesis: final-layer question-reading heads split into entity-name readers (shared by distractors) vs relation/wh-word readers (shared by bridge golds).
tt0 = list(toks[0]); s1 = sep_pos[0][0]; qtoks = tt0[1:s1]
print("question tokens:", " ".join(qtoks))
print("q0 paragraphs 0-7 gold:", practice[0]["gold"][:8], "| orig ranks:", F[F.qi == 0].rank0.values[:8].astype(int))
for L, hs in [(5, [1, 2, 3, 4, 5, 7, 8, 9, 10]), (4, [0, 2, 5, 6, 7, 9])]:
    for h in hs:
        a = np.mean([att[k, L, h, 0, 1:s1] for k in range(8)], 0)          # same question in all 8 inputs -> same positions
        top = np.argsort(-a)[:5]
        # variability across paragraphs of the attention to question span (does the head react to the paragraph?)
        qmass = [att[k, L, h, 0, 1:s1].sum() for k in range(8)]
        print("L%d h%-2d q-mass %.2f (range %.2f-%.2f) top q-tokens: %s" % (L, h, np.mean(qmass), min(qmass), max(qmass), ", ".join("%s:%.2f" % (qtoks[i], a[i]) for i in top)))
```