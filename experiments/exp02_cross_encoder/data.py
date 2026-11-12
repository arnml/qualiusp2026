"""Data access shared by train / score / analyze. Runs locally and on Colab.

The test (400, = BRACIS paper sample) and dev (1000) questions are fixed by ID in
split_ids.json (exported from exp01), so every experiment sees exactly the same questions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_IDS = json.loads((HERE / "split_ids.json").read_text())


def _load(split: str):
    from datasets import load_dataset

    # Local HF cache has "hotpot_qa"; the hub now also serves it as "hotpotqa/hotpot_qa".
    for name in ("hotpot_qa", "hotpotqa/hotpot_qa"):
        try:
            return load_dataset(name, "distractor", split=split)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def load_train(n: int | None = None, seed: int = 0) -> list[dict]:
    ds = _load("train")
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [dict(r) for r in ds]


def load_eval(which: str) -> list[dict]:
    """which in {'test', 'dev'}; returned in split_ids order."""
    ds = _load("validation")
    wanted = SPLIT_IDS[which]
    pos = {i: k for k, i in enumerate(wanted)}
    rows = [None] * len(wanted)
    for r in ds:
        k = pos.get(r["id"])
        if k is not None:
            rows[k] = dict(r)
    missing = sum(r is None for r in rows)
    if missing:
        raise RuntimeError(f"{missing} {which} ids not found in validation split")
    return rows


def paragraphs_of(row: dict) -> tuple[list[str], list[str]]:
    """Paper format: '<title>. <sentences joined>'."""
    titles, paras = [], []
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
        t = str(title).strip()
        body = " ".join(s.strip() for s in sents if s and s.strip())
        titles.append(t)
        paras.append(f"{t}. {body}" if body else t)
    return titles, paras


def gold_mask(row: dict) -> list[int]:
    gold = {str(t).strip() for t in row["supporting_facts"]["title"]}
    titles, _ = paragraphs_of(row)
    return [int(t in gold) for t in titles]


def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """Identical to cart-code experiments/preprocessing.py (BRACIS paper)."""
    context_str = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
    return f"""Answer the HotpotQA question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.
- If multiple positions/titles are mentioned, output only the one that
  directly answers the question.

Context:
{context_str}

Question: {question}

Answer:"""


_ENC = None


def n_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(text))


def results_dir() -> Path:
    """RESULTS_DIR env var (e.g. a Google Drive folder on Colab) or ./results."""
    p = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p
