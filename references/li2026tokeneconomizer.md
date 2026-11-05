```bibtex
@misc{li2026tokeneconomizer,
  title         = {{TokenEconomizer}: Experience-Driven Black-Box Cost Control for Coding Agents},
  author        = {Li, Aaron J. and Xu, Weijia and Singh, Chandan and Chen, Zixiao and Inala, Jeevana Priya and Siska, Charlotte and Tamersoy, Acar and Yu, Bin and Stoica, Ion and Galley, Michel and Gao, Jianfeng},
  year          = {2026},
  eprint        = {2610.22732},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2610.22732},
  url           = {https://arxiv.org/abs/2610.22732},
  pdf           = {https://arxiv.org/pdf/2610.22732},
  abstract      = {Advances in frontier coding agents have sharply increased the token costs of AI-assisted software engineering. Their token efficiency is typically evaluated on isolated tasks, whereas real-world developers use them to continuously build, update, and maintain large-scale projects. This mismatch motivates us to evaluate agents over sequences of realistic coding tasks, where we find that agents exhibit recurring environment-dependent token costs that existing methods struggle to reduce. To minimize these costs, we introduce TokenEconomizer, an external controller that cuts environment-specific token costs over a sequence of tasks. These cuts come from determining (i) which operational micro-skills to inject, (ii) which persistently irrelevant tools to prune, and (iii) when to intervene in unproductive failure loops. Across Claude Code, Codex, and GitHub Copilot CLI paired with frontier models, TokenEconomizer reduces the average token cost by 18.4-30.9\% with comparable task success. Savings hold in every repository under every tested agent-model setting and generalize to diverse domains such as sequential web development and security auditing, where costs fall by 6.4-21.2\% depending on task structure. Our work establishes adaptive black-box cost control as a promising direction for reducing token costs of coding agents in realistic settings.}
}
```

## TokenEconomizer: Experience-Driven Black-Box Cost Control for Coding Agents
- arXiv preprint, cs.CL, submitted 2026-10-15 (v1); notes below come from the abstract only, full text not yet read
- problem: token cost of coding agents is usually measured on isolated tasks, but real use is long sequences of tasks on evolving projects
- finding: over task sequences, agents show recurring environment-dependent token costs that existing methods struggle to reduce

proposal
- TokenEconomizer: external (black-box) controller, no access to agent internals
- three levers:
  - which operational micro-skills to inject
  - which persistently irrelevant tools to prune
  - when to intervene in unproductive failure loops

results
- tested with Claude Code, Codex, GitHub Copilot CLI + frontier models
- average token cost -18.4% to -30.9% with comparable task success; savings in every repo and every agent-model setting
- generalizes to sequential web development and security auditing: -6.4% to -21.2%
