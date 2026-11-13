OUTPUT:
```
Split: 120 selection questions; 80 confirmation questions
Paragraph counts: 1200 792
Initial edits: {'track': 'C', 'heads_removed': [], 'neurons_removed': 0, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}}
Initial cost: {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}
Tensor names and shapes:
bert.encoder.layer.0.attention.output.LayerNorm.bias (384,)
bert.encoder.layer.0.attention.output.LayerNorm.weight (384,)
bert.encoder.layer.0.attention.output.dense.bias (384,)
bert.encoder.layer.0.attention.output.dense.weight (384, 384)
bert.encoder.layer.0.attention.self.key.bias (384,)
bert.encoder.layer.0.attention.self.key.weight (384, 384)
bert.encoder.layer.0.attention.self.query.bias (384,)
bert.encoder.layer.0.attention.self.query.weight (384, 384)
bert.encoder.layer.0.attention.self.value.bias (384,)
bert.encoder.layer.0.attention.self.value.weight (384, 384)
bert.encoder.layer.0.intermediate.dense.bias (1536,)
bert.encoder.layer.0.intermediate.dense.weight (1536, 384)
bert.encoder.layer.0.output.LayerNorm.bias (384,)
bert.encoder.layer.0.output.LayerNorm.weight (384,)
bert.encoder.layer.0.output.dense.bias (384,)
bert.encoder.layer.0.output.dense.weight (384, 1536)
bert.pooler.dense.bias (384,)
bert.pooler.dense.weight (384, 384)
classifier.bias (1,)
classifier.weight (1, 384)

```
Passes left: 50/50. Turn 1/200.