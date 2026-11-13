# exp05: why does A1 still miss gold paragraphs?

**Goal:** a descriptive failure anatomy of the fine-tuned selector A1, to find what is left to improve (it feeds exp06 and exp07). No decision rules and no LLM.

**Data:** every HotpotQA validation question that is not in dev (1000) or test (400), the "pool". A1 never trained on validation questions. dev is analysed too, for comparison.

**Steps:**

```bash
uv sync
uv run python score_pool.py           # A1 scores + 384-d pooler features -> cache/ (git-ignored, ~1.5 h CPU)
uv run python analyze.py --split pool
uv run python analyze.py --split dev
```

**Failure:** both gold paragraphs are not in A1's top 3 (top 2 is also reported). Categories, first match wins:

| Category | Rule | Why it matters |
|---|---|---|
| A bridge-linked | the missed gold's title appears in the found gold's text, and not in the question | a chain-aware second hop (exp07) could recover it |
| B comparison | comparison question | both entities are usually named |
| C alt-evidence | a distractor ranked above the missed gold contains the answer | possible alternative evidence or an annotation issue |
| D other | — | — |

**Outputs:**
- `results/anatomy_<split>.json`: rates, categories and feature means for failures vs successes
- `results/failures_sample_<split>.md`: 20 random failures for manual reading

## Results

(pending)
