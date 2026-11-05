```bibtex
@misc{takehi2026weightspace,
  title        = {Sparks of Weight-Space Programming: {LLMs} can natively read \& write neural network weights},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = {nov},
  howpublished = {Blog post, MIT CSAIL},
  url          = {https://rikiyat.github.io/blog/weight-space-programming/}
}
```

## Sparks of Weight-Space Programming
- blog post (not peer reviewed), published 2026-11-02; all results are self-reported by the authors
- claim: frontier LLMs (they use "GPT-6 Astra") can read and write network weights directly, not through an optimizer / gradient descent

**Reading weights**
- identified the training task of four same-shaped PPO game networks from raw weights
- found the language and induction (copying) heads in small char-level transformers, no forward passes
- limit: could not spot the sycophantic Gemma 3 270M from weights alone; needed to inspect attention patterns on its own texts

**Writing / editing weights**
- fix: editing 3.5% of the weights of a Kinetix car-on-ramp policy gave success in 1,024/1,024 test episodes
- from scratch: MinAtar weights written in one reply beat PPO trained 100M steps (Breakout 517.4 vs 15.8)
- Go (1.5M params): 336-line program writing weights; beats smaller PPO nets with 0 forward passes, beats an equal-size PPO net with 50
- Qwen3.5-0.8B: reasoning loops traced to one direction in MLPs of layers 15 and 19; edited 131,072 weights (0.016%)
  - MATH-500 7.9% → 48.1%, GSM8K 13.8% → 67.1%, IFEval 21.0% → 62.3% (50 forward passes)
- limit: on Gemma 3 270M (rarely loops) no math gain → editing removes an obstacle, cannot add a missing ability

**Future directions listed:** backdoor auditing without evaluation, zero-data steering / unlearning, one-shot init, self-improvement, circuit grafting across architectures

**Cited / related:** Karpathy "Software 2.0"; Elhage et al. (induction heads); arXiv 2102.11174 (networks trained to write weights); arXiv 2301.05062 (compilers from formal programs); AdderBoard

**Relevance to us:** weights as a directly editable knowledge/behavior store, contrasts with retrieval (knowledge kept outside parameters, see REALM).
