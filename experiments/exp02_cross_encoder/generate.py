"""Answer the 400 test questions with the paper's generator, for each context condition.

  uv run python generate.py --limit 5     # pilot
  uv run python generate.py               # all 400 x all conditions (resumable)
  uv run python generate.py --report      # F1 / EM / tokens table only

Generator, prompt and F1 are the same as the BRACIS paper (gpt-5.4-mini, default params,
cart-code prompt, HotpotQA official normalisation). Requires OPENAI_API_KEY.
Answers are appended to results/generation/<condition>.jsonl (one line per question),
so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data import build_retrieval_prompt, gold_mask, load_eval, paragraphs_of, results_dir

MODEL = "gpt-5.4-mini"
PRICE_IN, PRICE_OUT = 0.20, 0.80  # USD per 1M tokens (cart-code baselines_config.py)


def conditions(rows, scores):
    """name -> {qid: list of paragraph indices, in the order shown to the LLM}."""
    def by_score(qid, keep):
        s = np.asarray(scores[qid])
        return [int(i) for i in np.argsort(-s) if i in keep]

    out = {"k10_all": {}, "oracle_gold": {}, "a1_top2": {}, "a1_t0.07": {}, "a1_t0.02": {}}
    for r in rows:
        q, s = r["id"], np.asarray(scores[r["id"]])
        p = 1 / (1 + np.exp(-s))
        out["k10_all"][q] = list(range(len(s)))  # dataset order, all 10
        out["oracle_gold"][q] = [i for i, g in enumerate(gold_mask(r)) if g]
        out["a1_top2"][q] = [int(i) for i in np.argsort(-s)[:2]]
        for t in (0.07, 0.02):  # thresholds chosen on dev for 0.90 / 0.95 both-gold
            keep = {int(i) for i in np.where(p >= t)[0]} or {int(np.argmax(s))}
            out[f"a1_t{t}"][q] = by_score(q, keep)
    return out


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_em(pred, gt):
    p, g = normalize(pred), normalize(gt)
    em = float(p == g)
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0, em
    pt, gtk = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gtk)).values())
    if not pt or not gtk or common == 0:
        return 0.0, em
    pr, rc = common / len(pt), common / len(gtk)
    return 2 * pr * rc / (pr + rc), em


def report(rows):
    gen = results_dir() / "generation"
    gt = {r["id"]: r["answer"] for r in rows}
    lines = []
    for f in sorted(gen.glob("*.jsonl")):
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        f1s, ems = zip(*(f1_em(r["answer"], gt[r["id"]]) for r in recs), strict=True)
        lines.append({
            "condition": f.stem, "n": len(recs), "f1": np.mean(f1s), "em": np.mean(ems),
            "input_tokens": np.mean([r["input_tokens"] for r in recs]),
            "usd_per_query": np.mean([r["cost_usd"] for r in recs]),
        })
    import pandas as pd

    df = pd.DataFrame(lines)
    df.to_csv(results_dir() / "generation_summary.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("\nPaper (same 400, gpt-5.4-mini): k=10 F1 0.792 @1452.5 tok; Noise-Gate tau=0.30 "
          "F1 0.764 @1065.3 tok; band = k10 F1 +-0.05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="subset of condition names")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows = load_eval("test")
    if args.report:
        return report(rows)

    from openai import OpenAI

    # Rate limits: the SDK retries 429/5xx with exponential backoff; give it room.
    client = OpenAI(max_retries=8, timeout=120)
    scores = json.loads((results_dir() / "scores_full.json").read_text())["test"]
    conds = conditions(rows, scores)
    gen = results_dir() / "generation"
    gen.mkdir(exist_ok=True)
    todo_rows = rows[: args.limit] if args.limit else rows

    for name, sel in conds.items():
        if args.only and name not in args.only:
            continue
        path = gen / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
        todo = [r for r in todo_rows if r["id"] not in done]
        print(f"{name}: {len(done)} done, {len(todo)} to run")
        lock = threading.Lock()

        def run(r, name=name, sel=sel, path=path, lock=lock):
            _, paras = paragraphs_of(r)
            idx = sel[r["id"]]
            prompt = build_retrieval_prompt(r["question"], [paras[i] for i in idx])
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}])
            u = resp.usage
            rec = {
                "id": r["id"], "condition": name, "paragraphs": idx,
                "answer": (resp.choices[0].message.content or "").strip(),
                "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                "cost_usd": (u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT) / 1e6,
            }
            with lock, open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        failed = 0
        with ThreadPoolExecutor(args.workers) as ex:
            for r, fut in [(r, ex.submit(run, r)) for r in todo]:
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001 - keep going; rerun fills gaps
                    failed += 1
                    print(f"  FAILED {name} {r['id']}: {type(e).__name__}: {str(e)[:120]}")
        if failed:
            print(f"{name}: {failed} failed -> rerun generate.py to retry only those")
    report(rows)


if __name__ == "__main__":
    main()
