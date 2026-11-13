```bibtex
@inproceedings{wu2024reft,
  title     = {{ReFT}: Representation Finetuning for Language Models},
  author    = {Wu, Zhengxuan and Arora, Aryaman and Wang, Zheng and Geiger, Atticus and Jurafsky, Dan and Manning, Christopher D. and Potts, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2404.03592}
}
```

## ReFT: Representation Finetuning for Language Models
- arXiv v1 2024-04-04 (cs.CL), latest v3 2024-05-22; the page lists no venue (bib says NeurIPS 2024); notes from the abstract only, full text not yet read

**Method**
- keep the base model frozen; learn task-specific interventions on hidden representations
- main instance: LoReFT (low-rank linear subspace ReFT), plus an ablation that trades some performance for efficiency
- presented as a drop-in replacement for existing PEFT methods; a training library is released

**Findings**
- interventions are 15x-65x more parameter-efficient than LoRA
- evaluated on eight commonsense reasoning tasks, four arithmetic reasoning tasks, instruction tuning, and GLUE
- best efficiency-performance balance; almost always outperforms state-of-the-art PEFTs

**Caveats**
- abstract gives no per-task numbers; the models used are not named on the page

**Relevance to us:** option for a hybrid edit: freeze A0 and fit a tiny representation-level intervention, with an LLM choosing layers or positions. A cheap alternative to editing weights directly.
