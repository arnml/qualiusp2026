The final edit saves **13.4% of encoder FLOPs**, with no original practice success lost.

- **Removed:** 96 low-strength MLP neurons from each layer 0–4, plus all 1,536 MLP neurons in layer 5: **2,016 total**. No heads, whole layers, or scalings.
- **Evidence:** Early removals targeted small incoming/outgoing weight norms. The final MLP proved unnecessary for practice rankings; removing the entire final layer instead sharply reduced quality, showing its attention remains necessary.
- **Quality:** Selection improved from **111/120 to 112/120**. Confirmation stayed **78/80**, with identical successful questions. Overall: **94.5% → 95.0%**.
- **Final cost:** **33.997824 GFLOPs/question**, **9,096,480 encoder parameters**.

Confirmation rejected larger cuts and guided rollback, so it is not an untouched final estimate. Quality on the unseen 1,000 questions remains unmeasured.