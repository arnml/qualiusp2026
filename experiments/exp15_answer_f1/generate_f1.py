"""exp15 (PROTOCOL.md): gpt-5.4-mini answer F1 with different contexts on exp13's fresh dev questions.

  uv run python generate_f1.py --limit 2      # smoke test (results/smoke/)
  uv run python generate_f1.py                # all pre-registered calls, interleaved per question (resumable)
  uv run python generate_f1.py --report       # pre-registered analysis -> results/results.json

Reader, F1/EM and normalisation are exp02's (loaded by path); F1/EM = max over answer and answer_aliases.
Requires OPENAI_API_KEY.
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

from data import build_retrieval_prompt, gold_mask, hops, load, paragraphs_of

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
# *_idx: the same top-5 sets in dataset-index order (primary, order-controlled); *_top5: ranker order (secondary)
CONDS = ("ctrl_idx", "iter_idx", "ctrl_top5", "iter_top5", "gold", "all")
REPEAT = ("ctrl_idx", "iter_idx")  # a second reader draw on >=3-hop questions
B = 5000


def contexts(rows):
    t = json.loads((HERE / "results" / "top5_dev_fresh.json").read_text())
    top = {}
    for c in ("ctrl_oneshot", "iter_iter"):
        s5 = t[c]["top5"]
        assert len(s5) == 5 * len(rows) and t[c]["first"] == rows[0]["id"] and t[c]["last"] == rows[-1]["id"], c
        top[c] = [[ord(ch) - 97 for ch in s5[5 * i:5 * i + 5]] for i in range(len(rows))]
        assert all(len(set(o)) == 5 and all(0 <= j < len(paragraphs_of(r)[1]) for j in o) for o, r in zip(top[c], rows)), c
    return {"ctrl_idx": [sorted(o) for o in top["ctrl_oneshot"]], "iter_idx": [sorted(o) for o in top["iter_iter"]],
            "ctrl_top5": top["ctrl_oneshot"], "iter_top5": top["iter_iter"],
            "gold": [[i for i, g in enumerate(gold_mask(r)) if g] for r in rows],
            "all": [list(range(len(paragraphs_of(r)[1]))) for r in rows]}


def score(pred, row, k):
    return max(g2.f1_em(pred, a)[k] for a in [row["answer"], *row.get("answer_aliases", [])])


def records(gen, name, rows, ctx):
    """Answers for one condition; rejects duplicates and records whose condition/paragraphs do not match."""
    recs, pos = {}, {r["id"]: i for i, r in enumerate(rows)}
    for j in map(json.loads, (gen / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()):
        assert j["id"] not in recs, f"duplicate {name} {j['id']}"
        assert j["condition"] == name and j["paragraphs"] == ctx[name.replace("_rep", "")][pos[j["id"]]], f"bad record {name} {j['id']}"
        recs[j["id"]] = j
    return recs


def boot(d, rng):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.percentile(bs, 5))


def report(rows, gen):
    rng, rep, F = np.random.default_rng(0), {}, {}
    ctx = contexts(rows); hp = np.array([hops(r) for r in rows]); long_ = hp >= 3
    for c in CONDS:
        recs = records(gen, c, rows, ctx)
        assert set(recs) == {r["id"] for r in rows}, f"{c}: missing answers"
        F[c] = np.array([score(recs[r["id"]]["answer"], r, 0) for r in rows])
        tok = np.array([recs[r["id"]]["input_tokens"] for r in rows])
        rep[c] = {"f1": round(float(F[c].mean()), 4), "em": round(float(np.mean([score(recs[r["id"]]["answer"], r, 1) for r in rows])), 4),
                  "blank_answers": int(sum(not recs[r["id"]]["answer"] for r in rows)),
                  "input_tokens_mean_p10_p90": [round(float(tok.mean()), 1), float(np.percentile(tok, 10)), float(np.percentile(tok, 90))],
                  "cost_usd": round(sum(j["cost_usd"] for j in recs.values()), 4),
                  "by_hops_f1": {int(h): round(float(F[c][hp == h].mean()), 4) for h in (2, 3, 4)}}
    Fa = {}
    for c in REPEAT:  # primary: order-controlled sets, >=3-hop, mean of the two reader draws
        rr = records(gen, c + "_rep", rows, ctx)
        assert set(rr) == {r["id"] for r, l in zip(rows, long_) if l}, f"{c}_rep: missing answers"
        second = np.array([score(rr[r["id"]]["answer"], r, 0) for r, l in zip(rows, long_) if l])
        Fa[c] = (F[c][long_] + second) / 2
        rep[c]["reader_draw_agreement_ge3hop"] = round(float(np.mean(F[c][long_] == second)), 4)
    d = Fa["iter_idx"] - Fa["ctrl_idx"]; lo, hi, _ = boot(d, rng)
    rep["primary_ge3hop_order_controlled"] = {"n": int(long_.sum()), "diff": round(float(d.mean()), 4), "ci95": [lo, hi], "success": bool(lo > 0)}
    dd = (F["iter_top5"] - F["ctrl_top5"])[long_]; lo, hi, _ = boot(dd, rng)
    rep["secondary_ge3hop_native_order"] = {"diff": round(float(dd.mean()), 4), "ci95": [lo, hi]}
    dd = F["iter_idx"] - F["ctrl_idx"]; lo, hi, _ = boot(dd, rng)
    rep["secondary_all_order_controlled"] = {"n": len(dd), "diff": round(float(dd.mean()), 4), "ci95": [lo, hi]}
    lo, hi, one = boot(dd[hp == 2], rng)
    rep["secondary_2hop_noninferiority"] = {"n": int((hp == 2).sum()), "diff": round(float(dd[hp == 2].mean()), 4), "ci95": [lo, hi],
                                            "one_sided_95_lower": one, "noninferior": bool(one > -0.02)}
    rep["label"] = "secondaries exploratory; the >=3-hop primary subgroup was chosen from exp13 recall on this same set"
    (HERE / "results" / "results.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    rows = load("dev_fresh")
    gen = HERE / "results" / ("smoke" if args.limit else "generation")
    if args.report:
        return report(rows, gen)
    from openai import OpenAI

    client = OpenAI(max_retries=8, timeout=120)
    ctx = contexts(rows); gen.mkdir(parents=True, exist_ok=True); hp = [hops(r) for r in rows]
    names = list(CONDS) + [c + "_rep" for c in REPEAT]
    done = {n: ({json.loads(x)["id"] for x in (gen / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
                if (gen / f"{n}.jsonl").exists() else set()) for n in names}
    jobs = [(n, i, r) for i, r in enumerate(rows[: args.limit or None]) for n in names  # interleaved per question
            if r["id"] not in done[n] and not (n.endswith("_rep") and hp[i] < 3)]
    print({n: len(d) for n, d in done.items()}, "done;", len(jobs), "calls to run", flush=True)
    lock = threading.Lock()

    def run(job):
        name, i, r = job
        _, paras = paragraphs_of(r)
        idx = ctx[name.replace("_rep", "")][i]
        resp = client.chat.completions.create(model=g2.MODEL, messages=[{"role": "user", "content": build_retrieval_prompt(r["question"], [paras[j] for j in idx])}])
        u = resp.usage
        rec = {"id": r["id"], "condition": name, "paragraphs": idx, "answer": (resp.choices[0].message.content or "").strip(),
               "model": resp.model, "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
               "cost_usd": (u.prompt_tokens * g2.PRICE_IN + u.completion_tokens * g2.PRICE_OUT) / 1e6}
        with lock, open(gen / f"{name}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for job, fut in [(j, ex.submit(run, j)) for j in jobs]:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - a rerun fills gaps
                failed += 1
                print(f"  FAILED {job[0]} {job[2]['id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"done ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
