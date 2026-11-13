OUTPUT:
```
105
['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.word_embeddings.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.dense.bias', 'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.self.key.bias', 'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.query.weight', 'bert.encoder.layer.0.attention.self.value.bias', 'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.intermediate.dense.bias', 'bert.encoder.layer.0.intermediate.dense.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.dense.bias', 'bert.encoder.layer.0.output.dense.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
{'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}
200 dict_keys(['id', 'question', 'answer', 'type', 'paragraphs', 'gold'])
Counter({10: 199, 2: 1})
Counter({'bridge': 163, 'comparison': 37})
1992 total items
(items=None, pairs=None, model: 'str' = 'edited', cls: 'bool' = False, mlp: 'bool' = False, attn_n: 'int' = 0) -> 'dict'
Run the network. ONE call = ONE forward pass of the budget (any size up to the limits).

items : list of (question_index, paragraph_index) into `practice` (<= 2048)
pairs : list of (question_text, paragraph_text) you write yourself (<= 64)
model : "edited" | "original" | "both"
cls   : also return [CLS] hidden state after embeddings and after every kept layer
mlp   : also return MLP activations (after GELU) at [CLS] for every kept layer
attn_n: also return attention probabilities for the first attn_n inputs (<= 8)
Returns numpy arrays: scores_<model>, and cls_<model>, mlp_<model>, attn_<model>, tokens.

tok len mean/median/max 147.70833333333334 136.0 512
{'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}
head {'encoder_gflops_per_question': 39.0578944, 'encoder_params': 10597536, 'removed': {'heads': 1, 'neurons': 0, 'layers': []}}
neuron {'encoder_gflops_per_question': 39.259392, 'encoder_params': 10646015, 'removed': {'heads': 0, 'neurons': 1, 'layers': []}}
layer {'encoder_gflops_per_question': 32.8018944, 'encoder_params': 8872320, 'removed': {'heads': 0, 'neurons': 0, 'layers': [5]}}
{'track': 'C', 'heads_removed': [], 'neurons_removed': 0, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}}

```
Passes left: 5/5. Turn 1/40.