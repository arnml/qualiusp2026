# exp15: does the long-chain recall gain reach the answers?

Pre-registered in [PROTOCOL.md](PROTOCOL.md) (commit 61caae6, revised after an adversarial Codex review).
- **Setup:** gpt-5.4-mini reads top-5 contexts from exp13's frozen models on the 1,017 unused MuSiQue dev questions.
- **Calls:** 7,078, none failed; cost $1.29.
- **Results:** [`results/results.json`](results/results.json); answers in `results/generation/`.

## Bottom line

1. **Primary holds: on ≥ 3-hop questions the better evidence set gives better answers.** Same five-paragraph budget, passages in index order so that only *which* passages are included differs, mean of two reader draws:
   - F1(conditioning-trained top 5) − F1(one-shot top 5) = **+0.076 [+0.041, +0.112]** (n = 488) ✅
2. **Secondary (exploratory):**
   - all questions: +0.038 [+0.011, +0.064]
   - 2-hop: non-inferior (+0.014; one-sided 95% lower bound −0.015 > −0.02)
   - in the rankers' own order: +0.057 on ≥ 3-hop
3. **Selection beats sending everything.** Every top-5 context scores higher F1 than all ~20 paragraphs (0.506), with **69% fewer input tokens** (about 790 vs 2,525). Distractors hurt the reader, as on HotpotQA (exp02).
4. **Large headroom remains:** gold paragraphs reach F1 0.716 with 463 tokens.

## Numbers (F1 / EM / mean input tokens)

| Context | F1 | EM | Tokens | F1 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|---|---|
| all ~20 paragraphs | 0.506 | 0.394 | 2,525 | 0.500 | 0.518 | 0.502 |
| one-shot top 5, index order | 0.538 | 0.438 | 793 | 0.543 | 0.566 | 0.465 |
| conditioning-trained top 5, index order | **0.576** | **0.475** | 783 | 0.558 | **0.625** | **0.540** |
| one-shot top 5, ranker order | 0.536 | 0.434 | 793 | 0.541 | 0.568 | 0.455 |
| conditioning-trained top 5, ranker order | 0.584 | 0.473 | 783 | 0.583 | 0.626 | 0.509 |
| gold paragraphs | 0.716 | 0.603 | 463 | 0.720 | 0.736 | 0.661 |

- **Reader stability:** the two reader draws gave the same F1 on 83–84% of ≥ 3-hop questions.
- **Caveat:** the ≥ 3-hop subgroup was chosen from exp13's recall results on this same question set (disclosed in PROTOCOL). The claim is "on these unused questions, the recall gain reaches the answers".
