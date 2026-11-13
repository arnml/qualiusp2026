"""MuSiQue-Ans loading, paragraph format and prompt (exp02 format; see PROTOCOL.md)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"dev": "musique_ans_v1.0_dev.jsonl", "pool": "musique_ans_v1.0_train.jsonl"}
SPLITS = {"dev": "dev", "test": "dev", "pool": "pool"}  # split -> source file


@lru_cache(maxsize=None)
def _file(src: str) -> dict:
    with open(HERE / "data" / FILES[src], encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def load(split: str) -> list[dict]:
    ids = json.loads((HERE / "split_ids.json").read_text())[split]
    rows = _file(SPLITS[split])
    return [rows[i] for i in ids]


def paragraphs_of(row: dict) -> tuple[list[str], list[str]]:
    """'<title>. <text>' as in exp02."""
    ps = sorted(row["paragraphs"], key=lambda p: p["idx"])
    titles = [p["title"].strip() for p in ps]
    return titles, [f"{t}. {p['paragraph_text'].strip()}" for t, p in zip(titles, ps)]


def gold_mask(row: dict) -> list[int]:
    return [int(p["is_supporting"]) for p in sorted(row["paragraphs"], key=lambda p: p["idx"])]


def hops(row: dict) -> int:
    return len(row["question_decomposition"])


def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """exp02/BRACIS prompt; only change: 'the HotpotQA question' -> 'the question'."""
    context_str = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
    return f"""Answer the question using only the provided context.

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
