```bibtex
@inproceedings{java2026frugalrag,
  title     = {{FrugalRAG}: Less Is More in {RL} Finetuning for Multi-Hop Question Answering},
  author    = {Java, Abhinav and Koundinyan, Srivathsan and Natarajan, Nagarajan and Sharma, Amit},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2507.07634}
}
```

## FrugalRAG: Less Is More in RL Finetuning for Multi-Hop QA
- arXiv v1 2025-07-10, latest v3 2026-02-28 (cs.CL); the page lists no venue (bib says ICLR 2026); notes from the abstract only, full text not yet read

**Setup / Method**
- argument: final-answer-reward RL, which helped small LMs on math and code, gives limited gains on multi-hop RAG
- two stages: (1) supervised finetuning of a small model on a full-exploration policy that generates broad sub-queries; (2) RL that adaptively reduces search depth by question difficulty, rewarding answers that are correct and efficient

**Findings**
- competitive performance with about 1,000 examples, versus roughly 10x more data for prior approaches
- on HotPotQA and other multi-hop benchmarks, improves the efficiency-accuracy tradeoff, cutting retrieval cost nearly in half
- generalizes zero-shot to BrowseCompPlus and beats SLM-based and other baselines

**Caveats**
- cost here is retrieval searches, not LLM prompt tokens as in our setup

**Relevance to us:** closest related work on adaptive cost for multi-hop QA. Supports exp07's idea of spending more (3-4 paragraphs) only on harder questions, though ours is a selector, not an RL search agent.
