```bibtex
@misc{vast2026mice,
  title         = {{MICE}: Minimal Interaction Cross-Encoders for efficient Re-ranking},
  author        = {Vast, Mathias and Morand, Victor and van Cooten, Basile and Soulier, Laure and Mothe, Josiane and Piwowarski, Benjamin},
  year          = {2026},
  eprint        = {2602.16299},
  archivePrefix = {arXiv},
  note          = {Accepted to EMNLP 2026 (main conference), per the arXiv comments; proceedings entry not yet checked}
}
```

## MICE: Minimal Interaction Cross-Encoders for efficient Re-ranking
- arXiv v1 2026-02-18, latest v4 2026-08-31; arXiv comments: "Accepted as a Main Conference paper to EMNLP 2026" (checked 2026-11-13); notes from the abstract only, full text not yet read

**Setup / Method**
- cross-encoders rank well but are costly, so they are mostly used as second-stage re-rankers
- prior work either sparsifies attention or strengthens first-stage retrieval (e.g. late interaction)
- MICE bridges the two: analyze cross-encoder internals, find and remove superfluous interactions

**Findings**
- keeps most of its cross-encoder counterparts' in-domain effectiveness
- matches or exceeds them out-of-domain
- reduces FLOPs "down to 2.5 times" (exact wording of the reduction not checked beyond the abstract)

**Caveats**
- datasets and base models are not given on the abstract page

**Relevance to us:** relevant to the later latency step: A0/A1 are cross-encoders, so pruning query-document interactions could cut compute beyond weight pruning. Not planned before exp06/exp07.
