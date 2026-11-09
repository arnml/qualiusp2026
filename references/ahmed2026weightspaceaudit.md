```bibtex
@misc{ahmed2026weightspaceaudit,
  title         = {Can an {LLM} Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring ``Weight-Space Programming''},
  author        = {Ahmed, Nazmus Sakib},
  year          = {2026},
  eprint        = {2611.08070},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.08070}
}
```

## Can an LLM Actually Reason and Program about Neural Network's Weights...?
- arXiv preprint, cs.LG, submitted 2026-11-05 (v1); notes from the abstract only, full text not yet read
- direct response to the weight-space programming claim (see `takehi2026weightspace.md`); audits the **reading** side only

**Setup**
- model tested: **Claude Opus 5.5**
- small CNNs, ~125k params, trained on five image datasets
- every weight given as plain text; no tools, no forward passes, no dataset or class names

**Findings**
- local arithmetic is accurate: infers ~3 optimizer steps from batch-norm statistics; recovers CIFAR-10's vehicle/animal split from output-layer sign patterns
- the model reads the first and last layers closely and samples the middle; several claims built on partial tracing fail verification
- semantic judgments lean on familiar datasets:
  - names CIFAR-10 for all three fast-food networks, relabels KMNIST as MNIST, reads a SAT-6 network as CIFAR-10 animals
  - permuting CIFAR-10 labels keeps the output split but every interpreted class pair becomes wrong → relies on remembered label order
- caveat: exploratory, mostly single runs; no direct forward-pass test
- conclusion: zero-shot weight reading = local arithmetic + dataset priors, not weight-flow tracing
- contribution: a protocol for auditing future weight-space claims

**Relevance to us:** counterpoint to the "LLM writes the gate weights" arm in `research_direction.md`. We only need weights that *work*, not understanding, but any weight-writing experiment should follow this audit protocol (permuted labels, held-out datasets, forward-pass verification).
