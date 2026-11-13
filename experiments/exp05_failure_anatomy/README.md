# exp05: why does A1 still miss gold paragraphs?

**Goal:** a descriptive failure anatomy of the fine-tuned selector A1, to find what is left to improve (it feeds exp06 and exp07). No decision rules and no LLM.

**Data:** a random sample (seed 0) of 4,000 HotpotQA validation questions not in dev (1000) or test (400), the "pool" (capped for CPU time). A1 never trained on validation questions. dev is analysed too, for comparison.

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

Data: `results/anatomy_pool.json` (4,000 unused validation questions), `results/anatomy_dev.json`, `results/failures_sample_pool.md` (20 random failures, read by hand).

### How often A1 fails

| Split | Both gold in top 3 | Both gold in top 2 | Top-3 failures |
|---|---|---|---|
| pool (4,000) | 0.953 | 0.877 | 189 |
| dev (1,000) | 0.959 | 0.877 | 41 |

- **Comparison questions never fail at top 3** (pool 804/804; top 2 0.99). **All top-3 failures are bridge questions** (bridge bg3 0.941).
- **Near misses:** the missed gold is at rank 4 in 50% of failures (95/189) and in ranks 4–5 in 71%.

### Why (failures vs successes, pool)

| Feature (about the lower-ranked gold, "missed") | Failures | Successes |
|---|---|---|
| its title appears in the question | **2.6%** | 33.9% |
| the other gold's title appears in the question | 30.2% | 54.4% |
| a distractor ranked above it has its title in the question | **27.0%** | 2.0% |
| a distractor ranked above it contains the answer | 22.2% | 1.3% |
| the found gold mentions its title (bridge link) | 33.3% | 35.5% |
| the same, ignoring "(…)" disambiguators in titles | 39.2% | 37.3% |
| word overlap (Jaccard) with the question | 0.104 | 0.111 |

**Categories of the 189 top-3 failures** (first match wins): A bridge-linked 61 (32%), C alt-evidence 32 (17%), D other 96 (51%), B comparison 0.

### Reading

1. **Failures are hidden-bridge questions.** The missed paragraph is about the bridge entity, which the question never names (its title is in the question 2.6% vs 34%). A1 ranks above it a distractor that *is* named in the question (27% vs 2%), or one that contains the answer string (22% vs 1%).
2. **The obvious chain feature does not separate failures.** The found gold mentions the missed gold's title just as often in successes (35–37%) as in failures (33–39%). A title-link feature alone can mark at most ~a third of failures, and it fires as often on questions A1 already gets right.
   - **Prior for exp07, stated before its results:** small gains at best from the text link features. Any gain must come from the pooled-vector pair interaction.
3. **Many "D other" failures are near-misses with an indirect link** that string matching misses:
   - *VAQ-135* vs *VAQ-129/142:* similar distractors.
   - *Hawaii Five-0 (2010 TV series):* the actor paragraph names the show, but the bridge is via "eighth season".
   - *Whistl* vs "Whistle": a spelling variant.
4. **Practical:** half of failures sit at rank 4. A cheap 4th paragraph only for uncertain questions (exp07's adaptive policy) targets exactly them.

### Limitations

- The categories use exact string matching (titles, answers), so they under-count links through aliases.
- The "(…)" variant is a post-hoc check, not pre-registered.
- A single model (A1) and one dataset.
