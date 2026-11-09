"""Shared helpers: data loading, embeddings cache, prompt, token counting, selectors.

Data loading, the 400-question sample, the paragraph format and the prompt are copied
from cart-code (BRACIS 2026 paper) so the numbers are comparable.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Same local snapshot the paper used (read-only).
DATA_DIR = Path(
    os.environ.get(
        "HOTPOT_DIR", r"C:\code\cart-code\experiments\cache\hotpot_qa\distractor\validation"
    )
)
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # as stated in the paper, 22M params, 384-d
N_TEST = 400  # paper sample: random.Random(42).sample(range(len(ds)), 400)
N_DEV = 1000  # disjoint validation questions, used only for threshold tuning


# --------------------------------------------------------------------------- data
def load_split() -> tuple[list[dict], list[dict]]:
    """Return (test, dev). test == the paper's 400 questions, dev is disjoint."""
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_DIR))
    test_idx = random.Random(42).sample(range(len(ds)), N_TEST)
    rest = sorted(set(range(len(ds))) - set(test_idx))
    dev_idx = random.Random(7).sample(rest, N_DEV)
    return [dict(ds[i]) for i in test_idx], [dict(ds[i]) for i in dev_idx]


def paragraphs_of(row: dict) -> tuple[list[str], list[str]]:
    """Paper format: '<title>. <sentences joined>'. Returns (titles, paragraphs)."""
    titles, paras = [], []
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
        t = str(title).strip()
        body = " ".join(s.strip() for s in sents if s and s.strip())
        titles.append(t)
        paras.append(f"{t}. {body}" if body else t)
    return titles, paras


def gold_titles(row: dict) -> set[str]:
    return {str(t).strip() for t in row["supporting_facts"]["title"]}


# --------------------------------------------------------------------------- prompt
def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """Identical to cart-code experiments/preprocessing.py."""
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
    """o200k_base = gpt-5.x tokenizer family; same counting basis as the paper."""
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(text))


# --------------------------------------------------------------------------- embeddings
def embed_rows(rows: list[dict], tag: str) -> list[dict]:
    """Embed question, paragraphs and 'question + paragraph' strings; cached to .npz.

    The q+p embeddings are only needed by the 'concat' bridge variant.
    """
    path = CACHE / f"emb_{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return list(z["items"])

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    texts, spans = [], []
    for r in rows:
        _, paras = paragraphs_of(r)
        q = r["question"]
        start = len(texts)
        texts += [q] + paras + [f"{q} {p}" for p in paras]
        spans.append((start, len(paras)))
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    items = []
    for (s, n) in spans:
        items.append(
            {"q": vecs[s], "p": vecs[s + 1 : s + 1 + n], "qp": vecs[s + 1 + n : s + 1 + 2 * n]}
        )
    np.savez_compressed(path, items=np.array(items, dtype=object))
    return items


# --------------------------------------------------------------------------- selectors
def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def jaccard_matrix(paras: list[str]) -> np.ndarray:
    n = len(paras)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = jaccard(paras[i], paras[j])
    return m


def _redundancy_filter(
    order: list[int], paras: list[str], rho: float, jac: np.ndarray | None = None
) -> list[int]:
    kept: list[int] = []
    for i in order:
        if jac is not None:
            if any(jac[i, j] > rho for j in kept):
                continue
        elif any(jaccard(paras[i], paras[j]) > rho for j in kept):
            continue
        kept.append(i)
    return kept


def select_noise_gate(e: dict, paras: list[str], tau: float, rho: float = 0.65) -> list[int]:
    """Paper Algorithm 1: cosine(q,p) >= tau in rank order, then Jaccard guard."""
    sims = e["p"] @ e["q"]
    order = [int(i) for i in np.argsort(-sims) if sims[i] >= tau]
    return _redundancy_filter(order, paras, rho, e.get("jac"))


def select_topk(e: dict, k: int) -> list[int]:
    sims = e["p"] @ e["q"]
    return [int(i) for i in np.argsort(-sims)[:k]]


def select_hop_gate(
    e: dict,
    paras: list[str],
    tau1: float,
    tau2: float,
    variant: str = "sum",
    n_anchor: int = 1,
    rho: float = 0.65,
) -> list[int]:
    """Option 1: hop-aware Noise-Gate.

    Hop 1: keep paragraphs with cos(q,p) >= tau1 (strict threshold). The top-1 paragraph
           by cos(q,p) is always an anchor, so hop 2 always has a starting point.
    Hop 2: for every other paragraph p, bridge(p) = max over anchors a of
             sum    : cos(p, normalize(e_q + e_a))      -- 0 extra encoder passes
             concat : cos(p, enc("q a"))                 -- 1 extra pass per anchor
             pp     : cos(p, e_a)                        -- 0 extra passes
           keep p if bridge(p) >= tau2.
    Then the usual Jaccard redundancy guard. Output order: hop-1 by cos(q,p), then hop-2
    by bridge score.
    """
    sims = e["p"] @ e["q"]
    rank = [int(i) for i in np.argsort(-sims)]
    hop1 = [i for i in rank if sims[i] >= tau1]
    anchors = rank[:n_anchor] if not hop1 else hop1[: max(n_anchor, 1)]
    hop1_set = set(hop1) | set(anchors)

    bridge = np.full(len(paras), -1.0)
    for a in anchors:
        if variant == "sum":
            v = e["q"] + e["p"][a]
            v = v / np.linalg.norm(v)
        elif variant == "concat":
            v = e["qp"][a]
        elif variant == "pp":
            v = e["p"][a]
        else:
            raise ValueError(variant)
        bridge = np.maximum(bridge, e["p"] @ v)

    hop2 = [
        int(i) for i in np.argsort(-bridge) if int(i) not in hop1_set and bridge[i] >= tau2
    ]
    first = [i for i in rank if i in hop1_set]
    return _redundancy_filter(first + hop2, paras, rho, e.get("jac"))
