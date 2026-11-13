Key finding: at layer 6, the direction separating hard golds from the distractors that beat them is mostly anti-readout (cos −0.94, by construction, since those distractors outscore them). Its component orthogonal to the readout still separates them on held-out split B: hard gold beats the beating distractors in 87% of B questions, paired mean/sd 0.76. The overlap direction among distractors is 0.94 cosine with the readout, so the readout is essentially a lexical-match detector.

Two confounds to check before building on this:
- Hard golds are shorter than the distractors that beat them (96 vs 117 tokens), so the orthogonal direction may partly encode length.
- I need a rough strength scale.

The strength check is a statistic, not a network simulation: on split A only, I add β·(projection) to recorded scores to see whether any plausible gain exists before spending a pass.

```python
# Hypothesis: the readout-orthogonal hard-gold direction is content (answer-bearing), not just paragraph length; a modest weight on it would improve A metrics.
hp = np.load(WORK / "hard_dir_L6_perp.npy")
ntc = cq(F.ntok.values.astype(float)); hac = cq(F.has_ans.values.astype(float)); tic = cq(F.title_in_q.values)
cands = {"hard_perp": hp, "ans_perp": a_o}
for nm, v in cands.items():
    p = X6raw @ v; pc = cq(p)
    print("%-10s corr(within-q): ntok=%.2f has_ans=%.2f title_in_q=%.2f overlap=%.2f score=%.2f | cos(hard_perp)=%.2f"
          % (nm, np.corrcoef(pc, ntc)[0,1], np.corrcoef(pc, hac)[0,1], np.corrcoef(pc, tic)[0,1], np.corrcoef(pc, ovc)[0,1], np.corrcoef(pc, s0c)[0,1], v @ hp))
# statistic-only strength scan on split A: s + beta * sd(score) * standardized projection (no network computation)
sdS = s0c.std(); FA, FB = F[A], F[~A]
rows = []
for nm, v in cands.items():
    p = X6raw @ v; z = (p - p.mean()) / cq(p).std()
    for beta in [0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        sA = S0[A] + beta * sdS * z[A]; mA = metrics(FA, sA)[0]
        rows.append((nm, beta, round(mA["b3"], 3), round(mA["tokens"], 1)))
print(pd.DataFrame(rows, columns=["dir", "beta", "A_b3", "A_tokens90"]).to_string(index=False))
```