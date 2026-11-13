## TASK

You are given a small text-relevance model, a cross-encoder with 22.7M parameters. Your job is to make it **better at one task by editing its weights directly**: find out what goes wrong inside the network, and repair it with targeted changes to the weights.

The model reads one question and one paragraph and outputs a single relevance score; higher means more relevant. Architecture: BERT, 6 layers, width 384, 12 attention heads of size 32 per layer, GELU MLP of width 1536 per layer, vocabulary 30,522.

```
input = [CLS] question [SEP] paragraph [SEP]      (token_type 0 = question part, 1 = paragraph part)
h     = 6 transformer layers (post-LayerNorm BERT)
score = classifier( tanh( pooler.dense( h[CLS] ) ) )
```

It was trained to rank web passages for search queries (MS MARCO). Here it is used for something different: **multi-hop question answering**. Each question comes with 10 Wikipedia paragraphs. Exactly 2 are needed to answer it (the "gold" paragraphs). The other 8 are distractors, retrieved because they share words with the question. Often one gold paragraph is about an entity that the question does not name directly (a "bridge" paragraph). The model should score both gold paragraphs above the distractors.

Today it does this poorly. On held-out questions, both gold paragraphs are in its top 3 only about 59% of the time. To keep both gold paragraphs for 90% of questions with a score threshold, it must keep about 1,090 prompt tokens of paragraphs per question, most of the ~1,250 available. The selected paragraphs are what a large language model reads, so every distractor that is kept costs tokens.

## HOW IT WILL BE MEASURED

After the session, your edited model scores held-out questions you never see, and so does the unedited model.

- **Main score, tokens at 90%:** for each question, the paragraphs whose score passes a threshold are kept (at least one). The threshold is set on held-out questions so that 90% of questions keep both gold paragraphs. The score is the average number of prompt tokens kept per question: **fewer tokens at the same 90% is better.**
- **Also reported:** the fraction of questions with both gold paragraphs among the 3 highest-scoring paragraphs ("both-gold@3").

Because the threshold is set afterwards, shifting or rescaling all scores changes nothing. Only the **ranking** of a question's gold paragraphs above its distractors matters.

## WHAT COUNTS

- **Edits must come from understanding the network:** a diagnosed mechanism and a targeted change to it. Examples of mechanisms:
  - a direction that tracks word overlap with the question
  - a head that ignores the paragraph's link to the question
  - a component that suppresses bridge paragraphs
- **In scope:** `bert.encoder.layer.*` and `bert.pooler.dense.*`.
- **Out of scope** (refused when you save an edit):
  - the embeddings: `bert.embeddings.*`
  - the output layer: `classifier.*`
  - any edit that only multiplies a whole tensor by a constant
  - any edit that changes more than 108,000 weights in total, about 1% of the encoder, counted against the original
- **Global knobs that shift or rescale all scores are not wanted.** The threshold is chosen afterwards, so they cannot help.

## WHAT YOU HAVE

1. **A persistent Python kernel** (CPU, no network). Libraries: numpy, scipy, scikit-learn, torch (CPU), pandas. Already imported:
   - **Reading:**
     - `NAMES`: tensor names
     - `load(name)`: the current tensor with your edits applied, float32 torch
     - `load_original(name)`
     - `tok`: the tokenizer
   - **`practice`:** {N} training questions. Each is a dict with `question`, `paragraphs` (usually 10 strings, "Title. text"; a few questions have fewer), `gold` (one 0/1 per paragraph), `answer` and `type` ("bridge" or "comparison").
   - **Editing:**
     - `save_edit(name, tensor)`: replaces a tensor with your edited version (checked against WHAT COUNTS)
     - `drop_edit(name)`: returns one tensor to the original
     - `reset_all()`
     - `edited()`: what is changed now, with the number of changed entries
   - **Folders:** the variable `WORK` is your read-write folder (a `pathlib.Path`; write files as `WORK / "name"`). Keep your notes in the file `NOTES` (= `WORK / "notes.md"`). Do not write anywhere else; absolute paths like `/work` do not exist here. Your saved edits are always the submitted edit; keep copies of good versions in `WORK`.
2. **`forward(...)`: runs the network; budget {B} passes in total.** One call is one pass, whatever its size; failed calls are not counted.
   - **Inputs:** `items=[(question_index, paragraph_index), ...]` from `practice` (up to 2,048, so one pass can score every practice paragraph), or `pairs=[(question_text, paragraph_text), ...]` you write yourself (up to 64).
   - **`model`:** `"edited"`, `"original"` or `"both"` (the same inputs through both models; counts as one pass).
   - **Optional outputs:**
     - `cls=True`: the [CLS] vector after the embeddings and after every layer
     - `mlp=True`: MLP activations at [CLS] for every layer
     - `attn_n=k`: attention probabilities for the first k ≤ 8 inputs, with their tokens
   - **Returns** numpy arrays (`scores_edited`, `cls_original`, …) and `passes_left`.

## HOW TO WORK

- **Diagnose before editing.** Find which practice questions the original model gets wrong, and what the distractors that outrank gold paragraphs have in common.
- **Localize causally.** Test a hypothesis about a component by changing it at several strengths, and check that the effect grows with strength.
- **Measure with enough power.** Over n questions, both-gold@3 has a standard error of about 50/√n points; with {N} questions that's about 3.5 points. Compare edited and original **on the same items** (`model="both"`), and look at per-question changes, not only averages.
- **Do not overfit the practice set.** Choose edits on one part of the practice questions, and confirm them on the other part. Say how you split. Keep an edit only if its gain is clearly larger than its standard error on the questions you did not use to choose it.
- **Passes are scarce:** {B} in total. Plan what each one must tell you. One pass with `model="both"` and `cls=True` or `mlp=True` on all practice paragraphs is very informative.
- **Write down your reasoning.** Start each code block with a one-line hypothesis. Do not print huge arrays. Keep your notes file updated.

## RULES

- **No gradient descent or training of any kind:** no backward passes, autograd, optimizers or iterative fitting to a loss.
- **No solving a regression or classification objective on the practice data** (least squares, logistic regression, SVM and similar) and writing the result into weights. Statistics you compute and reason about (means, differences of means, PCA/SVD, correlations) are allowed and encouraged; say which you used.
- **Do not run or simulate the network yourself:** no re-implementing its forward computation in numpy or torch. All model computation goes through `forward(...)`.
- **No internet. No other tools.** Do not access the harness's internals.
- **Code that breaks these rules is refused before it runs.**

## HOW THE SESSION RUNS

- Each reply may contain **one** Python code block. It runs in the kernel, and you see its output in the next turn.
- The session continues until all {B} forward passes are used, or for at most {TURN_CAP} turns. Replying without a code block does not end it while passes remain.
- When the budget is used up, you get up to 3 more turns to finalize your edit. Then reply **without a code block** with your final explanation:
  - which mechanism you found
  - what you changed (tensor names, rows/columns, formula, strength)
  - the evidence that it helps, from practice questions you did not use to choose the edit
  - `edited()` at the end
