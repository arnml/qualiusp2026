**No edit retained; the original weights are restored.**

Discovery analysis found an overlap-associated signal in layer 4: selected MLP neurons favored lexical distractors over weak gold paragraphs. This suggested a mechanism, but the interventions did not establish a reliable repair.

I tested 96 columns of `bert.encoder.layer.4.output.dense.weight` using  
\(W[:,S]=(1-\alpha)W_{\text{original}}[:,S]\), with strengths **0.5, 1.0, and 2.0**. Each candidate changed 36,864 weights. Column indices and reasoning are recorded in `WORK`.

Complete removal, α=1, performed best on the 100 discovery questions. On the **100 untouched confirmation questions**:

- Token-cost proxy at 90% retention: **962.67 → 936.71**.
- Saving: **25.96 tokens**, standard error **15.90**, 95% interval **−18.31 to +49.49**.
- Both-gold@3: **54% → 55%**, one win and zero losses.

The gain failed the predeclared confirmation criterion, so I reverted it. Notes and experiment results are saved.

Final `edited()` status: no heads, neurons, or layers removed; no scalings; **0 changed weights**.