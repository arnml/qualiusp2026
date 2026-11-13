```bibtex
@inproceedings{vasani2025phlora,
  title     = {{PHLoRA}: data-free Post-hoc Low-Rank Adapter extraction from full-rank checkpoint},
  author    = {Vasani, Bhoomit and FitzGerald, Jack and Fang, Anjie and Vaish, Sushmit},
  booktitle = {Findings of IJCNLP-AACL 2025},
  pages     = {1992--1999},
  year      = {2025},
  doi       = {10.18653/v1/2025.findings-ijcnlp.125}
}
```

## PHLoRA: data-free Post-hoc Low-Rank Adapter extraction
- arXiv v1 2025-09-13 (cs.LG); the page lists no venue (bib says Findings of IJCNLP-AACL 2025); notes from the abstract only, full text not yet read

**Method**
- decompose the weight difference between a base model and its fully fine-tuned version to extract a low-rank adapter (the SVD-of-delta idea)
- needs no training data and no gradients
- adapters can be merged, routed dynamically at inference (e.g. via S-LoRA), or served on platforms like NVIDIA NIM

**Findings**
- tested on text, image, and video benchmarks with the Amazon Nova model family
- authors say adapters keep most of the weight delta's energy, can be pruned safely, and cause negligible performance loss when re-merged

**Caveats**
- abstract gives no numbers or ranks; results are on large Nova models, not small encoders

**Relevance to us:** the low-rank counterpart to Panigrahi grafting in exp06: take the SVD of A1-A0, add back only rank-k parts to A0, and measure how much of A1's gain returns. Also gives a data-free way to ship A1 as a small adapter.
