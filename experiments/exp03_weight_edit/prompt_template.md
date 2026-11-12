# exp03 editor prompt (first turn), draft

Structure copied from Takehi et al. Exp 8 (TASK · HOW IT WILL BE MEASURED · WHAT COUNTS · WHAT YOU HAVE · HOW TO WORK · RULES · HOW THE SESSION RUNS). The wording is ours. Placeholders: `{N}`, `{B}`, `{TURN_CAP}`. Identical across editors and seeds.

---

## TASK

You are given a small text-relevance model, a cross-encoder with 22.7M parameters, and your job is to make it better at one task **by editing its weights directly**. Find out what goes wrong inside the network, and repair it with targeted changes to the weights.

The model reads one question and one paragraph and outputs a single number, a relevance score; higher means more relevant. Architecture: BERT, 6 layers, width 384, 12 attention heads of size 32, GELU MLP of width 1536, vocabulary 30,522.

```
input = [CLS] question [SEP] paragraph [SEP]      (token_type 0 = question part, 1 = paragraph part)
h     = 6 transformer layers (post-LayerNorm BERT)
score = classifier( tanh( pooler.dense( h[CLS] ) ) )
```

It was trained to rank web passages for search queries. Here it is used for something different: **multi-hop questions**. Each question comes with 10 Wikipedia paragraphs. Exactly 2 are needed to answer it (the "gold" paragraphs). The other 8 are distractors retrieved because they share words with the question. Often one gold paragraph is about an entity that the question does not name directly (a "bridge" paragraph). The model should give both gold paragraphs higher scores than the distractors.

## HOW IT WILL BE MEASURED

After the session, your edited model scores held-out questions you never see. For each question, the paragraphs whose score passes a threshold are kept. The threshold is fixed on a separate held-out set, so that 90% of questions keep both gold paragraphs. The result is the average number of tokens kept per question: **fewer tokens at the same 90% is better.** The same procedure is applied to the unedited model. Only weight values change; tensor names and shapes stay the same.

## WHAT COUNTS

- Edits must come from understanding the network: a diagnosed mechanism and a targeted change to it. Examples of mechanisms: a direction that tracks word overlap with the question, a head that ignores the paragraph's link to the question, a component that suppresses bridge paragraphs.
- In scope: `bert.encoder.layer.*` and `bert.pooler.dense.*`.
- Out of scope, and refused by `save_edit`:
  - the embeddings: `bert.embeddings.*`
  - the output layer: `classifier.*`
  - any edit that only multiplies a whole tensor by a constant
  - any edit that changes more than 108,000 weights in total, counted against the original
- Global knobs that shift or rescale all scores are not wanted. The measurement uses a threshold chosen afterwards, so they cannot help.

## WHAT YOU HAVE

1. **A persistent Python kernel** (CPU, no network). Libraries: numpy, scipy, scikit-learn, torch (CPU), pandas, tokenizers. Functions:
   - `NAMES`: list of tensor names
   - `load(name)`: the current tensor (your edit if saved, otherwise the original), as float32 torch
   - `load_original(name)`
   - `save_edit(name, tensor)`: checked against WHAT COUNTS
   - `drop_edit(name)`
   - `edited()`: names and the number of changed entries
   - `tok`: the tokenizer
   - `practice`: {N} training questions, each with its question, 10 paragraphs (title + text) and gold labels
   - Folders: `/work` (read-write; keep `/work/notes.md` updated), `/work/edited` (what is submitted)
2. **A forward-pass tool, budget {B} passes in total.** Call `forward(...)` in a code block. One call is one pass, whatever its size up to the limits; failed calls are not counted. Each call runs the model with your current edit, the original, or both on the same inputs (`model="edited" | "original" | "both"`). Inputs:
   - practice items (`items=[(q_index, p_index), ...]`, up to 2,048 pairs), or
   - texts you write yourself (`pairs=[(question, paragraph), ...]`, up to 64).
   It returns scores, and optionally the [CLS] hidden state after each layer, MLP activations at [CLS], and attention patterns for up to 8 pairs.

## HOW TO WORK

- **Diagnose before editing.** Find which practice questions the original model gets wrong, and what the distractors that outrank gold paragraphs have in common.
- **Localize causally.** Test a hypothesis about a component by changing it at several strengths and checking that the effect grows with strength.
- **Measure with enough power.** The rate of questions with both gold paragraphs in the top 3 has a standard error of about 50/√n points over n questions.
- **Compare on the same items.** Use `model="both"` so the original and the edit are compared on identical items.
- **Keep an edit only if** its gain is clearly larger than its standard error on practice questions you did **not** use to choose it. Split the practice set yourself and say how.
- Start each code block with a one-line hypothesis. Do not print long token lists. Keep copies of good edits in `/work`.

## RULES

- **No gradient descent or training of any kind:** no backward passes, autograd, optimizers, or iterative fitting to a loss.
- **No solving a regression or classification objective on the practice data** (least squares, logistic regression, SVM and similar) and writing the result into weights. Statistics you compute and reason about (means, differences of means, PCA/SVD, correlations) are allowed and encouraged; say which you used.
- **Do not run or simulate the network yourself;** all model computation goes through `forward(...)`.
- **No internet.** No other tools.
- Allowed: any analysis of the weights and of forward-pass results, and any computed edit that stays in scope.

## HOW THE SESSION RUNS

- Each of your replies may contain one Python code block. It runs in the kernel, and you see its output in the next turn.
- The session continues until all {B} forward passes are used, or for at most {TURN_CAP} turns. Replying without a code block does not end it.
- At the end, the tensors in `/work/edited` are submitted. Your final reply must explain:
  - which mechanism you found
  - what you changed (tensor names, rows/columns, formula, strength)
  - the evidence that it helps, from practice questions not used to choose the edit
