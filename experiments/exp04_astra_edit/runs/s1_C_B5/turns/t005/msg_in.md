OUTPUT:
```
[forward] pass used; 4 left
Returned structure: {'scores_original': {'shape': (992,), 'dtype': 'float32'}, 'cls_original': {'shape': (992, 7, 384), 'dtype': 'float32'}, 'mlp_original': {'shape': (992, 6, 1536), 'dtype': 'float32'}, 'attn_original': {'shape': (8, 6, 12, 256, 256), 'dtype': 'object'}, 'attn_tokens_original': {'shape': (8, 256), 'dtype': 'object'}, 'passes_left': '4'}
Attention examples: [{'q': 0, 'p': 0, 'gold': 1, 'type': 'comparison', 'title_overlap': 0.5}, {'q': 0, 'p': 3, 'gold': 0, 'type': 'comparison', 'title_overlap': 0.333}, {'q': 2, 'p': 1, 'gold': 1, 'type': 'comparison', 'title_overlap': 1.0}, {'q': 2, 'p': 7, 'gold': 0, 'type': 'comparison', 'title_overlap': 0.5}, {'q': 3, 'p': 7, 'gold': 1, 'type': 'bridge', 'title_overlap': 0.6}, {'q': 3, 'p': 0, 'gold': 0, 'type': 'bridge', 'title_overlap': 0.8}, {'q': 4, 'p': 0, 'gold': 1, 'type': 'comparison', 'title_overlap': 0.333}, {'q': 4, 'p': 2, 'gold': 0, 'type': 'comparison', 'title_overlap': 0.429}]

```
Passes left: 4/5. Turn 4/40.