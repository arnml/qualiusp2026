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
