```bibtex
@inproceedings{yin2024lofit,
  title     = {{LoFiT}: Localized Fine-tuning on {LLM} Representations},
  author    = {Yin, Fangcong and Ye, Xi and Durrett, Greg},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2406.01563}
}
```

## LoFiT: Localized Fine-tuning on LLM Representations
- NeurIPS 2024 (camera ready), arXiv v1 2024-06-03 (cs.CL); notes from the abstract only, full text not yet read
- sits between interpretability-style interventions and fine-tuning

**Method**
- step 1: identify the attention heads most important for the task
- step 2: train offset vectors added to the hidden representations at only those heads

**Findings**
- localizes to a sparse set of heads, 3%-10%, and learns from limited training data
- on truthfulness and reasoning tasks, beats representation-intervention methods such as Inference-time Intervention
- head choice matters: heads picked for a different task do worse
- across 7 tasks, comparable to LoRA and other PEFT methods while modifying 20x-200x fewer parameters

**Caveats**
- abstract is about LLM attention heads; transfer to a small encoder cross-encoder is untested here

**Relevance to us:** model for a hybrid edit: a selector (Astra) chooses where to intervene, and a small fit sets the values. The head-specificity result supports letting the choice of location be task-specific.
