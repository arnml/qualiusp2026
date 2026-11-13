# exp07: chain-aware, adaptive evidence selector

- **Protocol:** `PROTOCOL.md` (pre-registered e6b7a7f).
- **Code:** `chain.py` (pair scorers, policies and statistics) and `generate_f1.py` (the follow-up).
- **Inputs:** A1 scores and pooler vectors cached by exp05 (`../exp05_failure_anatomy/cache`).

**Results files:**
- `results/chain_results_prereg.json`: the pre-registered run, the official numbers.
- `results/chain_results.json`: the same run plus the exploratory `a1only` ablation. The text/full point estimates are identical, but the 0.95 bootstrap CIs differ in the second decimal because the random draws shifted.
- `results/test_selections.json`: the per-question paragraph sets.

## Answer in one line

**Adaptive selection sends ~2–2.4 paragraphs instead of A1's threshold set, saving 57–71 prompt tokens per question (15–16%) on test, with no significant loss of recall.**
- Most of the saving at recall 0.90 comes from **adaptive sizing**.
- At 0.95, the **pair features add ~40 tokens** beyond what A1's own confidence gives.
- With exactly 2 paragraphs, the pair scorer does **not** pick the gold pair significantly more often than A1's top 2.

## Pre-registered results (test, 400 questions; thresholds tuned on dev)

### 1. Primary: tokens at recall 0.90 / 0.95 vs the A1 threshold

| Target | Policy | Test recall | Test tokens | Paragraphs | Δ tokens (95% CI) | Δ recall (95% CI) |
|---|---|---|---|---|---|---|
| 0.90 | A1 threshold | 0.915 | 384.0 | — | | |
| 0.90 | adaptive, text | 0.893 | 333.0 | 2.07 | −50.1 [−64.7, −36.4] | −0.020 [−0.050, +0.010] |
| 0.90 | **adaptive, full** | 0.890 | **326.2** | 2.02 | **−56.9 [−71.3, −43.6]** | −0.023 [−0.050, +0.005] |
| 0.95 | A1 threshold | 0.950 | 438.6 | — | | |
| 0.95 | adaptive, text | 0.935 | 378.0 | 2.43 | −59.8 [−80.1, −40.0] | −0.013 [−0.043, +0.015] |
| 0.95 | **adaptive, full** | 0.940 | **366.5** | 2.34 | **−71.2 [−90.7, −53.2]** | −0.008 [−0.035, +0.018] |

**Verdict: success**, for both scorers and both targets. Every token CI is below 0, and every recall CI includes 0.

**Caveat:** the adaptive policies land 1–2.5 points *below* the A1 threshold's realised test recall, though not significantly. The equal-recall comparison below (item 3) removes this advantage, and the gain remains.

### 2. Secondary: exactly 2 paragraphs, pair_best vs A1 top-2 (both gold kept)

| Scorer | pair_best | A1 top-2 | Only pair_best right / only top-2 right | McNemar p |
|---|---|---|---|---|
| text | 0.8775 | 0.8775 | 0 / 0 | 1.0 |
| full | 0.885 | 0.8775 | 3 / 0 | 0.25 |

**Verdict: not significant.** On dev, the full scorer fixed 18 vs 1, but on test only 3 vs 0. The pair scorer almost always agrees with A1's top 2. **Its value is in knowing when the top 2 is enough, not in picking a different pair.**

### 3. Descriptive: equal recall on test (each method's best threshold on test itself)

| Target | A1 threshold | adaptive, A1 scores only (exploratory) | adaptive, text | adaptive, full |
|---|---|---|---|---|
| 0.90 | 368.0 | 341.9 | 339.8 | **332.3** |
| 0.95 | 438.6 | 415.9 | 381.1 | **374.5** |

**Exploratory ablation** (not pre-registered; added to answer "is it chain awareness or just confidence?"):
- **`a1only`:** the same adaptive pipeline, but the pair scorer sees only A1's two scores, their difference and their ranks. So its best pair is always A1's top 2, and it decides only how many paragraphs to send.
- **At 0.90**, A1-only sizing gets most of the saving (−26 of −36 tokens).
- **At 0.95**, the text/pair features matter: −64 tokens with the full scorer vs −23 with A1-only.
- **Reading:** pair and link features help most where a 3rd or 4th paragraph is needed. That fits exp05: the misses are hidden-bridge paragraphs at rank 4.

### 4. Follow-up (triggered by 1): gpt-5.4-mini answer F1

(running; `results/f1_report.json`)

## Limitations

- **Dataset:** HotpotQA distractor only (10 candidate paragraphs). The stronger test of chain awareness is MuSiQue or 2Wiki.
- **Training data:** the pair scorer is trained on 3,200 pool questions, with A1 frozen.
- **Cost:** the pair scorer adds a small MLP over 6 pairs per question. Its latency was not measured; it is far smaller than A1's 10 cross-encoder passes.
- **Ablation:** the `a1only` ablation is exploratory, and the oracle frontier tunes on test (the same procedure for every method).
- **One model:** a single A1 checkpoint and a single train/validation split, with no seed variance.
