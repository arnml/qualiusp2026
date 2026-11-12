The final edit saves **23.84% of encoder FLOPs**. On the untouched 100-question confirmation half, both-gold@3 changed from **97% to 96%**, meeting the one-point practice tolerance.

- **Removed:** 3,584 MLP neurons—384, 512, 512, 512, 640, and 1,024 from layers 0–5 respectively. All heads and layers remain; no rescalings.
- **Evidence:** Neurons were ranked using `[CLS]` activation RMS and output-weight strength. Earlier layers also used input-weight strength to protect paths potentially useful at paragraph tokens. The final layer permits more pruning because only its `[CLS]` output reaches the scorer.
- **Causal checks:** Selection quality improved from 92% to 93%. Adding eight head removals or pruning more neurons reduced it to 88%; both edits were rejected.
- **Confirmation:** One regression and no gains. Across all 200 questions, quality remained **94.5%**, with two regressions and two gains.
- **Final cost:** **29.9034624 GFLOPs/question**, **7,890,688 encoder parameters**.

The chosen edit remains applied. Reasoning and exact removal indices are saved in `/work/notes.md` and `/work/removal_plan.json`. CPU timing and unseen-test quality remain unmeasured.