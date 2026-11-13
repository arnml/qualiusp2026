## TASK

You are given a small text-relevance model, a cross-encoder, already trained for one task. Your job is to make it **cheaper to run by removing parts of the network it does not need**, while keeping **all** of its quality on that task. You do this by **reading and editing its weights directly**: find which components carry the task and which are redundant, then remove the redundant ones.

The model reads one question and one paragraph and outputs a single relevance score; higher means more relevant. Architecture: BERT, 6 layers, width 384, 12 attention heads of size 32 per layer, GELU MLP of width 1536 per layer, vocabulary 30,522.

```
input = [CLS] question [SEP] paragraph [SEP]      (token_type 0 = question part, 1 = paragraph part)
h     = 6 transformer layers (post-LayerNorm BERT)
score = classifier( tanh( pooler.dense( h[CLS] ) ) )
```

The task is **multi-hop question answering**. Each question comes with 10 Wikipedia paragraphs. Exactly 2 are needed to answer it (the "gold" paragraphs). The other 8 are distractors retrieved because they share words with the question. Often one gold paragraph is about an entity the question does not name directly (a "bridge" paragraph). The model was fine-tuned for this and is good at it: on held-out questions, both gold paragraphs are in its top 3 about 94% of the time. It is used to decide which paragraphs a large language model gets to read, so it runs on every question, 10 times (once per paragraph), usually on a CPU. Today it costs about 39.26 GFLOPs per question in the encoder (10646784 encoder parameters).

## HOW IT WILL BE MEASURED

After the session, your edited model is run on held-out questions you never see, and so is the unedited model.

- **Quality:** the fraction of questions with both gold paragraphs among the 3 highest-scoring paragraphs ("both-gold@3"). Your edit must **not lose quality**: held-out both-gold@3 must stay within **0.5 percentage points** of the unedited model (measured on 1,000 held-out questions, about ±0.7 points of noise, so aim for no measurable loss on practice questions).
- **Cost (the main score):** encoder FLOPs per question, computed from what you removed (`cost()` in the kernel uses the same formula). **Lower is better, provided quality holds.** An edit that loses quality is scored by the FLOPs other methods need to reach the same lower quality, so a deep cut that loses 2 points is worth less than a moderate cut that loses nothing.
- **Also reported:** CPU time per question, measured after removed parts are physically deleted from the network.

Only removals and the allowed rescalings below change the model. Everything else stays identical.

## WHAT COUNTS

- **Allowed edits:** removing attention heads, removing MLP neurons, removing whole layers, and multiplying rows or columns of the remaining weight matrices by a constant to compensate for a removal. Removed parts are deleted, so they cost nothing.
- **Removals should come from understanding the network:** what a component does for this task, and evidence that the task does not need it. Examples: heads that attend only to [SEP] or punctuation, neurons that are never active at [CLS], a layer whose contribution to the [CLS] direction the classifier reads is negligible.
- **Out of scope:**
  - editing individual weight values in any other way (`save_edit` is not available in this track)
  - the embeddings
  - the classifier
- **A removal that hurts quality on practice questions is not free.** Removing a layer saves far more than removing a few neurons, but it also risks more.

## WHAT YOU HAVE

1. **A persistent Python kernel** (CPU, no network). Libraries: numpy, scipy, scikit-learn, torch (CPU), pandas. Already imported:
   - **Reading:**
     - `NAMES`: tensor names
     - `load(name)`: the current tensor with your removals applied, float32 torch
     - `load_original(name)`
     - `tok`: the tokenizer
   - **`practice`:** 200 questions the model was **not** trained on. Each is a dict with `question`, `paragraphs` (usually 10 strings, "Title. text"; a few questions have fewer), `gold` (one 0/1 per paragraph), `answer` and `type`.
   - **Removing:**
     - `remove_head(layer, head)`, `remove_neurons(layer, indices)`, `remove_layer(layer)`
     - `restore_head`, `restore_neurons`, `restore_layer`
     - `scale_rows(name, rows, f)`, `scale_cols(name, cols, f)`, `drop_scalings()`
     - `reset_all()`
   - **Inspecting:**
     - `edited()`: what is removed now
     - `cost()`: FLOPs and parameters now
   - **Folders:** the variable `WORK` is your read-write folder (a `pathlib.Path`; write files as `WORK / "name"`). Keep your notes in the file `NOTES` (= `WORK / "notes.md"`). Do not write anywhere else; absolute paths like `/work` do not exist here. Your current removals are always the submitted edit.
2. **`forward(...)`: runs the network; budget 50 passes in total.** One call is one pass, whatever its size; failed calls are not counted.
   - **Inputs:** `items=[(question_index, paragraph_index), ...]` from `practice` (up to 2,048, so one pass can score every practice paragraph), or `pairs=[(question_text, paragraph_text), ...]` you write yourself (up to 64).
   - **`model`:** `"edited"`, `"original"` or `"both"` (the same inputs through both models; counts as one pass).
   - **Optional outputs:**
     - `cls=True`: the [CLS] vector after the embeddings and after every kept layer
     - `mlp=True`: MLP activations at [CLS] for every kept layer
     - `attn_n=k`: attention probabilities for the first k ≤ 8 inputs, with their tokens
   - **Returns** numpy arrays (`scores_edited`, `cls_original`, …) and `passes_left`.

## HOW TO WORK

- **Diagnose before removing.** Use the weights and a few well-chosen passes to find out which heads, neurons and layers matter for scoring gold above distractors.
- **Localize causally.** Remove a candidate set, measure the change in both-gold@3 on practice questions, and prefer evidence over intuition.
- **Measure with enough power.** Over n questions, both-gold@3 has a standard error of about 50/√n points; with 200 questions that's about 3.5 points. Compare edited and original **on the same items** (`model="both"`), and look at per-question changes, not only averages.
- **Do not overfit the practice set.** Choose removals on one part of the practice questions, and confirm them on the other part. Say how you split.
- **Passes are scarce:** 50 in total. Plan what each one must tell you. One pass with `model="both"` and `cls=True` or `mlp=True` on all practice paragraphs is very informative.
- **Write down your reasoning.** Start each code block with a one-line hypothesis. Do not print huge arrays. Keep your notes file updated.

## RULES

- **No gradient descent or training of any kind:** no backward passes, autograd, optimizers or iterative fitting to a loss.
- **No solving a regression or classification objective on the practice data** (least squares, logistic regression and similar) to decide weights. Statistics you compute and reason about (means, variances, correlations, PCA/SVD, importance from your own removal experiments) are allowed; say which you used.
- **Do not run or simulate the network yourself:** no re-implementing its forward computation in numpy or torch. All model computation goes through `forward(...)`.
- **No internet. No other tools.** Do not access the harness's internals.
- **Code that breaks these rules is refused before it runs.**

## HOW THE SESSION RUNS

- Each reply may contain **one** Python code block. It runs in the kernel, and you see its output in the next turn.
- The session continues until all 50 forward passes are used, or for at most 200 turns. Replying without a code block does not end it while passes remain.
- When the budget is used up, you get up to 3 more turns to finalize your removals. Then reply **without a code block** with your final explanation:
  - what you removed
  - the mechanism or evidence for each removal
  - the measured effect on practice questions you did not use to choose it
  - the final `cost()`
