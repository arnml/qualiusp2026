# AI use log (for venue disclosure)

| Tool | Role | Where |
|---|---|---|
| Claude Code (Claude Opus 5.x) | Research agent: protocol drafts, code, running experiments, analysis, write-ups. Every protocol and claim is reviewed by the student and PI | all experiments |
| Claude subagents | Literature search and positioning; methodology review | `LITERATURE_POSITIONING.md`, `reviews/AI_ASSISTED_RESEARCH_PRACTICES.md` |
| Codex CLI (OpenAI, gpt-6-luna / gpt-6-astra) | Independent reviewer (`reviews/codex_review.md`); LLM weight editor "Astra" in exp04 | exp04, reviews |
| gpt-5.4-mini (API) | Reader LLM for answer F1 | exp02, exp07 |

**Rules:**
- **Citations:** a citation enters the thesis only after its arXiv, ACL or DOI page has been opened and checked.
- **Reviews:** AI reviews are inputs, not authority. Disagreements are resolved by the student and PI.
