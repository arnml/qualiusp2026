```bibtex
@inproceedings{panigrahi2023skilllocalization,
  title     = {Task-Specific Skill Localization in Fine-tuned Language Models},
  author    = {Panigrahi, Abhishek and Saunshi, Nikunj and Zhao, Haoyu and Arora, Sanjeev},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {202},
  pages     = {27011--27033},
  year      = {2023},
  note      = {arXiv:2302.06600}
}
```

## Task-Specific Skill Localization in Fine-tuned Language Models
- ICML 2023, arXiv v1 2023-02-13 (cs.CL); notes from the abstract only, full text not yet read
- introduces "skill localization": finding where a fine-tuned model's new skill lives in the weights

**Method**
- a simple optimization picks a tiny subset of parameters
- **grafting**: copy only the fine-tuned values for that subset onto the pre-trained model; no further training

**Findings**
- subset is about 0.01% of parameters yet accounts for over 95% of the fine-tuned model's performance
- grafted model performs almost as well as the fully fine-tuned one
- reports better in-distribution calibration (40-90% error reduction vs vanilla fine-tuning) and better out-of-distribution prediction quality
- in multi-task models, the sparse regions for different tasks are almost disjoint; overlap acts as a task-similarity proxy
- authors suggest grafting-based localization may help some continual-learning settings

**Caveats**
- abstract does not say which models or tasks; unknown whether it transfers to a 22.7M cross-encoder

**Relevance to us:** direct template for exp06: graft a sparse part of the A1-A0 weight delta into A0 and see how much of A1's gain survives. It is also the baseline against which an Astra-chosen graft set can be compared.
