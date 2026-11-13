# The go/no-go probe, explained simply

**In one line:** before paying a big AI to repair a small model, we spent ~5 minutes of CPU checking whether there is anything inside the small model to repair. There isn't much.

## The idea in plain words

Think of the small model (the 22.7M cross-encoder) as a student reading a question and a paragraph, then saying how useful the paragraph is.

The weight-editing idea (exp03) assumes the student **already knows the answer deep down, but something blocks it from saying so**. The Qwen example in the weight-space blog was like that: the model knew how to solve math, but got stuck repeating itself. Astra removed the "repeat" habit and the knowledge came out.

So before running the expensive experiment, we asked: **does our student secretly know which paragraphs are the gold ones?**

## How we checked (the "probe")

1. We gave the model 5,000 question–paragraph pairs: 200 practice questions plus 300 held-out questions, 10 paragraphs each.
2. For each pair, we recorded the model's **internal state**, the 384 numbers it computes at the [CLS] position, after each of its 6 layers. That's like recording the student's thoughts at every step of reasoning, not only their final answer.
3. Then we asked, layer by layer: **can a simple straight line in those numbers separate gold paragraphs from distractors?** We tried two ways:
   - **Average difference:** the average "thought" for gold paragraphs minus the average for distractors. This is the kind of simple statistic the AI editor would be allowed to use.
   - **Linear probe:** a small logistic regression trained on the thoughts. The editor is *not* allowed this. It is our "best possible" check of what's linearly hidden in there.
4. We compared both with what the model actually says, its own output score.

If the thoughts separated gold from distractors **much better than the output does**, the knowledge would be there but blocked, and an edit could unblock it. That would be a **go**.

## What we found

| What we read | How well it separates gold from distractors (AUC, 0.5 = coin flip, 1.0 = perfect) | Both gold in top 3 |
|---|---|---|
| The model's own answer | 0.825 | 58% |
| Simple average-difference direction (what the editor may use) | 0.824 | 58% |
| Best linear probe, any layer (upper bound) | 0.886 | 65% |
| The same model after normal training on just 200 questions | — | 78% |
| The same model after full training (90k questions) | — | 94% |

- **The simple direction just repeats what the model already says** (correlation 0.985). Nothing hidden.
- **Even the best probe gains only +7 points.** Ordinary training on 200 questions already does much better (+20 points).
- **Conclusion (as first written):** the student doesn't secretly know the answer *in a form a straight-line readout can find*. ~~The skill of spotting multi-hop evidence is missing, not blocked.~~
  **Correction (2026-11-13, PI):** a failed linear probe does not show the capability is absent from A0. It shows only that *this* probe (linear, [CLS], one paragraph at a time) could not extract it reliably. The information may be non-linear, distributed across layers or tokens, or recoverable only through interactions between paragraphs.
- **What the probe does support:** there is no single unused [CLS] direction to unlock. Removing one blockage won't help, so the expensive editing experiment is unlikely to beat normal training. **No-go**, decided by a rule written down *before* we looked.

## Why this methodology is useful (reusable recipe)

This is a cheap way to answer "is it worth trying X on this model?" before spending money or GPU time.

1. **State the hypothesis as something measurable inside the model.** Here: "a direction in the hidden states separates gold from distractors better than the output."
2. **Write the go/no-go rule before looking** (we committed it in `PROTOCOL.md`, commit `f0f955a`). This keeps us honest.
3. **Run the model once, recording internal states.** Forward passes only, no training; a laptop CPU is enough for a 22M model (~5 minutes for 5,000 pairs).
4. **Compare three readouts:**
   - what the model says (its output)
   - what a *simple, allowed* operation can extract
   - what the *best linear readout* can extract (the upper bound)
   The gaps between them tell you where the headroom is.
5. **Put the numbers next to a cheap baseline** (here, small-data fine-tuning). If the best possible readout is already below the cheap baseline, stop.
6. **Note the limits.** Here: only the [CLS] position, only linear directions, 300 held-out questions. Attention-level mechanisms could still hold surprises.

This is called **linear probing**, a standard tool in interpretability research. We used it as a **decision gate** for an experiment.

## What it means for the thesis

- A short, honest section: *"Why direct weight editing is not the right tool for this selector"*, with the table above.
- It strengthens exp02: a small model has to be **trained** to gain multi-hop relevance; it cannot be **repaired** into it.
- exp03 stays a good PhD question in a setting where a failure mode exists. For example, editing an already fine-tuned selector to fix its remaining mistakes.

Files: `probe.py` (the code), `results/probe.json` (all numbers), `PROTOCOL.md` §9b (formal verdict).
