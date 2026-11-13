"""exp05: why does A1 still miss gold paragraphs? Descriptive anatomy on the unused validation pool.

  uv run python analyze.py --split pool       # main analysis (needs cache/a1_pool.npz)
  uv run python analyze.py --split dev

Failure = the two gold paragraphs are not both in A1's top 3 (also reported: top 2).
For each failure, "missed gold" = the lower-ranked gold paragraph, "found gold" = the other one.
Categories (first match wins):
  A bridge-linked : the missed gold's title appears in the found gold's text, and not in the question
                    (a chain-aware second hop could recover it)
  B comparison    : comparison question (both entities usually named in the question)
  C alt-evidence  : a distractor ranked above the missed gold contains the answer string
                    (possible alternative evidence or annotation issue; answer not yes/no)
  D other
Base rates of the same features over all questions are reported for comparison.
Writes results/anatomy_<split>.json and results/failures_sample_<split>.md (20 random failures).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from data import SPLIT_IDS, _load, gold_mask, paragraphs_of

HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on and or to for is was were be by with as at from which what who whom whose when where "
           "how that this these those it its did does do are has have had than both same".split())


def words(s):
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def jac(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


def norm(s):
    return " ".join(WORD.findall(s.lower()))


def contains(text, s):
    s = norm(s)
    return bool(s) and f" {s} " in f" {norm(text)} "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(HERE / "cache" / f"a1_{args.split}.npz")
    rows = {r["id"]: dict(r) for r in _load("validation")}
    owner, score = z["owner"], z["score"].astype(float)
    recs = []
    for k, qid in enumerate(z["ids"]):
        r = rows[str(qid)]
        titles, paras = paragraphs_of(r)
        g = np.array(gold_mask(r), bool)
        s = score[owner == k]
        if g.sum() != 2 or len(s) != len(paras):
            continue
        order = np.argsort(-s)
        rank = np.empty(len(s), int); rank[order] = np.arange(1, len(s) + 1)
        gi = np.where(g)[0]
        miss, found = (gi[0], gi[1]) if rank[gi[0]] > rank[gi[1]] else (gi[1], gi[0])
        ans = str(r["answer"])
        yn = norm(ans) in {"yes", "no"}
        above = [i for i in order[: rank[miss] - 1] if not g[i]]
        rec = {"id": str(qid), "type": r["type"], "fail3": bool(rank[miss] > 3), "fail2": bool(rank[miss] > 2),
               "miss_rank": int(rank[miss]), "n_paras": len(paras),
               "miss_title_in_q": contains(r["question"], titles[miss]),
               "found_title_in_q": contains(r["question"], titles[found]),
               "link_found_to_miss": contains(paras[found], titles[miss]),
               "link_miss_to_found": contains(paras[miss], titles[found]),
               "jac_q_miss": jac(r["question"], paras[miss]), "jac_q_found": jac(r["question"], paras[found]),
               "jac_q_top_distractor": jac(r["question"], paras[above[0]]) if above else None,
               "answer_in_miss": (not yn) and contains(paras[miss], ans),
               "answer_in_found": (not yn) and contains(paras[found], ans),
               "distractor_above_has_answer": (not yn) and any(contains(paras[i], ans) for i in above),
               "distractor_above_title_in_q": any(contains(r["question"], titles[i]) for i in above)}
        if rec["link_found_to_miss"] and not rec["miss_title_in_q"]:
            rec["category"] = "A bridge-linked"
        elif r["type"] == "comparison":
            rec["category"] = "B comparison"
        elif rec["distractor_above_has_answer"]:
            rec["category"] = "C alt-evidence"
        else:
            rec["category"] = "D other"
        rec["_text"] = {"q": r["question"], "answer": ans, "titles": titles,
                        "top3": [titles[i] for i in order[:3]], "miss_title": titles[miss],
                        "found_title": titles[found], "miss_para": paras[miss][:400],
                        "top_distractor": paras[above[0]][:300] if above else ""}
        recs.append(rec)

    def rate(rs, key):
        v = [x[key] for x in rs if x[key] is not None]
        return round(float(np.mean(v)), 3) if v else None

    feats = ["miss_title_in_q", "found_title_in_q", "link_found_to_miss", "link_miss_to_found", "answer_in_miss",
             "answer_in_found", "distractor_above_has_answer", "distractor_above_title_in_q",
             "jac_q_miss", "jac_q_found", "jac_q_top_distractor"]
    fail = [x for x in recs if x["fail3"]]
    ok = [x for x in recs if not x["fail3"]]
    out = {"split": args.split, "questions": len(recs),
           "bg3": round(1 - len(fail) / len(recs), 4), "bg2": round(1 - np.mean([x["fail2"] for x in recs]), 4),
           "failures_top3": len(fail),
           "by_type": {t: {"n": sum(x["type"] == t for x in recs),
                           "bg3": round(1 - np.mean([x["fail3"] for x in recs if x["type"] == t]), 4),
                           "bg2": round(1 - np.mean([x["fail2"] for x in recs if x["type"] == t]), 4)}
                       for t in ("bridge", "comparison")},
           "categories_top3": dict(Counter(x["category"] for x in fail).most_common()),
           "categories_top2_only": dict(Counter(x["category"] for x in recs if x["fail2"] and not x["fail3"]).most_common()),
           "miss_rank_top3_failures": dict(sorted(Counter(x["miss_rank"] for x in fail).items())),
           "features": {f: {"failures": rate(fail, f), "successes": rate(ok, f)} for f in feats}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"anatomy_{args.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    rng = np.random.default_rng(args.seed)
    lines = [f"# 20 random top-3 failures of A1 ({args.split}, seed {args.seed})\n"]
    for x in rng.choice(fail, min(20, len(fail)), replace=False):
        t = x["_text"]
        lines += [f"## {x['id']} | {x['type']} | category {x['category']} | missed gold at rank {x['miss_rank']}",
                  f"- **Q:** {t['q']}", f"- **Answer:** {t['answer']}",
                  f"- **Gold:** found `{t['found_title']}`, missed `{t['miss_title']}`",
                  f"- **A1 top 3:** {', '.join(f'`{a}`' for a in t['top3'])}",
                  f"- **Missed gold (start):** {t['miss_para']}",
                  f"- **Highest distractor above it (start):** {t['top_distractor']}", ""]
    (HERE / "results" / f"failures_sample_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
