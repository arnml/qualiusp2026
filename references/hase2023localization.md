```bibtex
@inproceedings{hase2023localization,
  title     = {Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models},
  author    = {Hase, Peter and Bansal, Mohit and Kim, Been and Ghandeharioun, Asma},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {arXiv:2301.04213}
}
```

## Does Localization Inform Editing?
- NeurIPS 2023 (spotlight), arXiv v1 2023-01-10 (cs.LG); notes from the abstract only, full text not yet read
- tests whether "where a fact is stored" (localization) tells you where to edit it

**Findings**
- a fact can be changed by editing weights in a different place than localization methods such as Causal Tracing predict
- those methods are widely used to choose which mid-layer MLP weights to edit, so the practice is questioned
- also studied related tasks (erasing and amplifying facts): for one, editing performance relates to localization results, but the choice of layer predicts performance much better
- conclusion: better mechanistic understanding does not always give better ways to change behavior

**Caveats**
- about factual knowledge in language models, not rankers or encoders

**Relevance to us:** caution for exp06 and the Astra edits: a "where did fine-tuning change things" map (delta magnitude, grafting) may not be where an edit works best. Compare localized edits against simple layer-level baselines.
