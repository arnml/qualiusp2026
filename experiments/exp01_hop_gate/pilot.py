"""LLM pilot with stateless subagents (one question per subagent, no memory, no tools).

  uv run python pilot.py make  <selector_key> [n=5]   -> results/pilot_prompts.json
  uv run python pilot.py score                          -> needs results/pilot_answers.json

<selector_key> is a key of results/test_selections.json, e.g. "hop@0.9".
Questions are chosen where the hop gate keeps both gold paragraphs and Noise-Gate
(tau=0.30) does not, i.e. where option 1 should matter. Ties broken by fewer tokens.
"""

from __future__ import annotations

import csv
import json
import re
import string
import sys
from collections import Counter

from common import RESULTS, build_retrieval_prompt, gold_titles, load_split, n_tokens, paragraphs_of

PAPER_CSV = {
    "noise_gate": r"C:\code\cart-code\experiments\results\cart\results_noise_gate_gpt-5.4-mini.csv",
    "baseline": r"C:\code\cart-code\experiments\results\baseline\baseline_gpt-5.4-mini.csv",
}


# HotpotQA official normalisation / F1 (same as cart-code eval_utils.py)
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gt: str) -> float:
    p, g = normalize(pred), normalize(gt)
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0
    pt, gtk = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gtk)).values())
    if not pt or not gtk or common == 0:
        return 0.0
    pr, rc = common / len(pt), common / len(gtk)
    return 2 * pr * rc / (pr + rc)


def paper_rows():
    out: dict[tuple[str, str], dict] = {}
    for kind, path in PAPER_CSV.items():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = r["method"] if kind == "baseline" else f"noise_gate_{r['threshold']}"
                out[(r["question_id"], key)] = r
    return out


def make(key: str, n: int = 5):
    test, _ = load_split()
    sels = json.load(open(RESULTS / "test_selections.json"))
    hop = sels[key]["sel"]
    ng = sels["noise_gate_tau0.30"]["sel"]
    cands = []
    for r in test:
        titles, paras = paragraphs_of(r)
        gold = {i for i, t in enumerate(titles) if t in gold_titles(r)}
        h, g = hop[r["id"]], ng[r["id"]]
        if gold <= set(h) and not gold <= set(g):
            prompt = build_retrieval_prompt(r["question"], [paras[i] for i in h])
            cands.append((n_tokens(prompt), r, h, g, prompt, gold))
    cands.sort(key=lambda c: c[0])
    print(f"{len(cands)} test questions where {key} recovers gold and noise_gate@0.30 misses")
    picked = []
    for tok, r, h, g, prompt, gold in cands[:n]:
        picked.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "hop_sel": h,
                "ng_sel": g,
                "gold_idx": sorted(gold),
                "prompt_tokens": tok,
                "prompt": prompt,
            }
        )
    json.dump({"selector": key, "items": picked}, open(RESULTS / "pilot_prompts.json", "w"), indent=1)
    for p in picked:
        print(p["id"], p["prompt_tokens"], "tok |", p["question"], "->", p["answer"])


def score():
    pp = json.load(open(RESULTS / "pilot_prompts.json"))
    ans = json.load(open(RESULTS / "pilot_answers.json"))
    paper = paper_rows()
    lines = []
    for it in pp["items"]:
        a = ans[it["id"]]
        k10 = paper.get((it["id"], "retrieval_k10"), {})
        ng = paper.get((it["id"], "noise_gate_0.3"), {})
        lines.append(
            {
                "id": it["id"],
                "question": it["question"],
                "gold": it["answer"],
                "haiku_hop": a,
                "f1_haiku_hop": round(f1(a, it["answer"]), 3),
                "tok_hop": it["prompt_tokens"],
                "paper_k10_pred": k10.get("answer_pred"),
                "paper_k10_f1": k10.get("f1"),
                "paper_k10_tok": k10.get("input_tokens"),
                "paper_ng030_pred": ng.get("answer_pred"),
                "paper_ng030_f1": ng.get("f1"),
                "paper_ng030_tok": ng.get("input_tokens"),
            }
        )
    json.dump(lines, open(RESULTS / "pilot_scored.json", "w"), indent=1)
    for line in lines:
        print(json.dumps(line, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    else:
        score()
