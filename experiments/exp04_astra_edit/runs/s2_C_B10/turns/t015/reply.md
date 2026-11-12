The edit saves **27.24% of encoder FLOPs**, but narrowly missed the quality target on independent confirmation.

- **Removed:** 4,096 MLP neurons—512, 512, 512, 640, 640, and 1,280 from layers 0–5. All heads and layers remain; no rescaling.
- **Evidence:** selected neurons using low weight magnitude and low `[CLS]` activation energy, while protecting strong gold–distractor contrasts. Discovery performance was 112/120 versus 111/120 originally. Harmful layer and head removals were restored.
- **Untouched confirmation:** 77/80 versus 78/80 originally—one paired loss, no gains, a **1.25-percentage-point decline**. This exceeds the requested one-point tolerance.
- **Final `cost()`:** **28.566528 GFLOPs/question**, **7,496,960 encoder parameters**.

The submitted edit remains frozen from before confirmation. Notes and exact removal indices are saved in `WORK`.