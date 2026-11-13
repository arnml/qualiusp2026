"""exp08 pre-registered follow-up (only if the primary succeeds): gpt-5.4-mini answer F1 on the 400 test questions.

  uv run python generate_f1.py --cond adaptive_full     # winner name; runs winner + a1_thresh at 0.9 and 0.95
  uv run python generate_f1.py --cond adaptive_full --report

Generator, F1 and normalisation are exp02's (loaded by path); F1 = max over answer and answer_aliases.
Answers go to results/generation/<condition>.jsonl (resumable). Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import build_retrieval_prompt, load, paragraphs_of

HERE = Path(__file__).resolve().parent
EXP02 = HERE.parent / "exp02_cross_encoder"
sys.path.insert(0, str(EXP02))
_saved = sys.modules.pop("data", None)
_spec = importlib.util.spec_from_file_location("exp02_generate", EXP02 / "generate.py")
g2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2)
sys.modules.pop("data", None)
if _saved is not None:
    sys.modules["data"] = _saved
sys.path.pop(0)
GEN = HERE / "results" / "generation"


def f1(pred, row):
    return max(g2.f1_em(pred, a)[0] for a in [row["answer"], *row.get("answer_aliases", [])])


def report(rows, conds):
    out, per = {}, {}
    by = {r["id"]: r for r in rows}
    for c in conds:
        f = GEN / f"{c}.jsonl"
        if not f.exists():
            continue
        recs = {j["id"]: j for j in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if j}
        per[c] = {q: f1(recs[q]["answer"], by[q]) for q in by if q in recs}
        out[c] = {"n": len(per[c]), "f1": round(float(np.mean(list(per[c].values()))), 4),
                  "input_tokens": round(float(np.mean([recs[q]["input_tokens"] for q in per[c]])), 1),
                  "cost_usd": round(sum(recs[q]["cost_usd"] for q in per[c]), 4)}
    rng = np.random.default_rng(0)
    for i in range(0, len(conds), 2):
        a, b = per.get(conds[i]), per.get(conds[i + 1])
        if a and b and set(a) == set(b):
            d = np.array([a[q] - b[q] for q in a])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            out[f"paired_f1_{conds[i]}_minus_{conds[i + 1]}"] = {
                "mean": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    (HERE / "results" / "f1_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="primary winner, e.g. adaptive_full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    conds = [f"{args.cond}_0.9", "a1_thresh_0.9", f"{args.cond}_0.95", "a1_thresh_0.95"]
    rows = load("test")
    if args.report:
        return report(rows, conds)
    from openai import OpenAI

    client = OpenAI(max_retries=8, timeout=120)
    sels = json.loads((HERE / "results" / "test_selections.json").read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    todo_rows = rows[: args.limit] if args.limit else rows
    for name in conds:
        sel, path = sels[name], GEN / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
        todo = [r for r in todo_rows if r["id"] not in done]
        print(f"{name}: {len(done)} done, {len(todo)} to run", flush=True)
        lock = threading.Lock()

        def run(r, name=name, sel=sel, path=path, lock=lock):
            _, paras = paragraphs_of(r)
            idx = sel[r["id"]]
            prompt = build_retrieval_prompt(r["question"], [paras[i] for i in idx])
            resp = client.chat.completions.create(model=g2.MODEL, messages=[{"role": "user", "content": prompt}])
            u = resp.usage
            rec = {"id": r["id"], "condition": name, "paragraphs": idx,
                   "answer": (resp.choices[0].message.content or "").strip(),
                   "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                   "cost_usd": (u.prompt_tokens * g2.PRICE_IN + u.completion_tokens * g2.PRICE_OUT) / 1e6}
            with lock, open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        failed = 0
        with ThreadPoolExecutor(args.workers) as ex:
            for r, fut in [(r, ex.submit(run, r)) for r in todo]:
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001 - keep going; a rerun fills gaps
                    failed += 1
                    print(f"  FAILED {name} {r['id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if failed:
            print(f"{name}: {failed} failed -> rerun to retry only those", flush=True)
    report(rows, conds)


if __name__ == "__main__":
    main()
