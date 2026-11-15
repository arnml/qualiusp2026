"""exp16 data: HotpotQA (distractor) and 2WikiMultiHopQA from pinned Hugging Face parquet files (PROTOCOL.md).

Rows are normalized to {"id", "question", "paras": ["<title>. <text>", ...], "gold": [0/1 per paragraph],
"chain": [gold positions in hop order], "type"}. Chain order = first appearance of each title in supporting_facts.
DATA_DIR holds hotpot_{train0,train1,validation}.parquet and twowiki_{train,dev}.parquet.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", HERE / "data"))
FILES = {"hp_eval": ["hotpot_validation.parquet"], "hp_train": ["hotpot_train0.parquet", "hotpot_train1.parquet"],
         "tw_eval": ["twowiki_dev.parquet"], "tw_train": ["twowiki_train.parquet"]}
N_TRAIN, TRAIN_SEED = 1000, 16


def _norm_hotpot(r):
    titles, sents = r["context"]["title"], r["context"]["sentences"]
    sf = list(dict.fromkeys(r["supporting_facts"]["title"]))
    return titles, ["".join(s).strip() for s in sents], sf, r["type"]


def _norm_twowiki(r):
    ctx = json.loads(r["context"]); sf = list(dict.fromkeys(s[0] for s in json.loads(r["supporting_facts"])))
    return [c[0] for c in ctx], [" ".join(c[1]).strip() for c in ctx], sf, r["type"]


def _row(r, ds):
    titles, texts, sf, typ = (_norm_hotpot if ds == "hp" else _norm_twowiki)(r)
    pos = {t: i for i, t in enumerate(titles)}
    chain = [pos[t] for t in sf if t in pos]
    assert len(chain) == len(sf), "supporting title missing from context"
    gold = [int(i in chain) for i in range(len(titles))]
    return {"id": r["id"] if ds == "hp" else r["_id"], "question": r["question"],
            "paras": [f"{t.strip()}. {x}" for t, x in zip(titles, texts)], "gold": gold, "chain": chain, "type": typ}


@lru_cache(maxsize=None)
def _table(key):
    rows = []
    for f in FILES[key]:
        rows += pq.read_table(DATA / f).to_pylist()
    ds = key[:2]
    return {(r["id"] if ds == "hp" else r["_id"]): r for r in rows}


def load(split):
    """split in {hp_test, hp_tune, tw_test, tw_tune, hp_train, tw_train}."""
    ds = split[:2]
    if split.endswith("train"):  # PROTOCOL: seeded sample of N_TRAIN sorted training ids
        tab = _table(f"{ds}_train"); ids = sorted(tab)
        ids = sorted(ids[i] for i in np.random.default_rng(TRAIN_SEED).permutation(len(ids))[:N_TRAIN])
    else:
        tab = _table(f"{ds}_eval"); ids = json.loads((HERE / "split_ids.json").read_text())[split]
    return [_row(tab[i], ds) for i in ids]


def query(question, selected):
    """Question + selected paragraphs, most recent first (as exp09-exp13)."""
    return " [SEP] ".join([question, *reversed(selected)])


def encode(tok, qs, ps, max_length=512):
    try:
        return tok(qs, ps, truncation="only_first", max_length=max_length, padding=True, return_tensors="pt")
    except Exception:  # noqa: BLE001 - paragraph alone exceeds max_length
        return tok(qs, ps, truncation="longest_first", max_length=max_length, padding=True, return_tensors="pt")
