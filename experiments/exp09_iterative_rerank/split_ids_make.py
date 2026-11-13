"""Fix exp09's training questions before any training: 2,000 MuSiQue-Ans train questions, disjoint from exp08's pool.
dev300 = the first 300 ids of exp08's dev; test = exp08's test (400). Writes split_ids.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E8 = json.loads((HERE.parent / "exp08_musique" / "split_ids.json").read_text())
with open(HERE.parent / "exp08_musique" / "data" / "musique_ans_v1.0_train.jsonl", encoding="utf-8") as fh:
    ids = [json.loads(x)["id"] for x in fh]
free = sorted(set(ids) - set(E8["pool"]))
train = sorted(np.random.default_rng(9).choice(free, 2000, replace=False).tolist())
(HERE / "split_ids.json").write_text(json.dumps({"train": train, "dev300": E8["dev"][:300], "test": E8["test"]}))
print(len(free), len(train))
