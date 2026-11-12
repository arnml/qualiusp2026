"""exp04 harness: an LLM editor (Astra via Codex, or a scripted dummy) edits the selector.

Modelled on Takehi et al. 2026 Exp 8 (RikiyaT/lm_understand-logs): one editor call per turn,
all built-in editor tools off, the editor's only instrument is a persistent Python kernel,
and model runs go through a budgeted forward-pass server.

  uv run python harness.py --run stage1_C --track C --budget 5 --turn-cap 40 --editor astra
  uv run python harness.py --run dry --track C --budget 2 --turn-cap 6 --editor dummy --script dummy_script.json

Everything is logged under runs/<run>/ (see README / DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

import editlib as el
from data import gold_mask, load_eval, load_train, paragraphs_of, SPLIT_IDS

HERE = Path(__file__).resolve().parent
OUT_CAP = 10_000
EXEC_TIMEOUT = 900
LIMIT_ITEMS, LIMIT_PAIRS, LIMIT_ATTN = 2048, 64, 8
HANDOFF_EVERY = 40

_FEATURES_OFF = ["shell_tool", "unified_exec", "unified_exec_tty", "apps", "browser_use",
                 "browser_use_external", "computer_use", "image_generation", "in_app_browser",
                 "multi_agent", "plugins", "view_image", "sleep_tool", "tool_suggest",
                 "skill_search", "hooks", "goals", "code_mode_host"]
CODEX_OFF = [x for f in _FEATURES_OFF for x in ("--disable", f)]
# Values without quotes: Codex falls back to the raw string when TOML parsing fails, which
# avoids Windows cmd.exe quoting problems.
CODEX_CFG = ["-c", "mcp_servers.node_repl.enabled=false", "-c", "mcp_servers.MCP_DOCKER.enabled=false",
             "-c", "web_search=disabled"]
CODEX_BIN = __import__("shutil").which("codex") or "codex"
ALLOWED_ITEMS = {"agent_message", "reasoning", "error"}

# Hard bans: refused before execution. Soft flags: executed, recorded for the audit.
HARD = [r"\.backward\s*\(", r"torch\.optim", r"autograd", r"requires_grad_\s*\(\s*True",
        r"AutoModel", r"BertModel", r"from\s+transformers\s+import\s+(?!AutoTokenizer)",
        r"\bsubprocess\b", r"\bsocket\b", r"urllib", r"requests\.", r"os\.system", r"editlib",
        r"_BASE\b", r"kernel_api\._", r"EDIT_DIR", r"open\([^)]*state\.json"]
SOFT = [r"lstsq", r"LogisticRegression", r"LinearRegression", r"Ridge", r"\bSVC\b", r"\.fit\s*\(",
        r"torch\.nn\.functional", r"F\.gelu", r"softmax", r"layer_norm"]


# ===================================================================== practice data
def build_practice(kind: str, n: int, seed: int = 0) -> list[dict]:
    """'heldval': n validation questions disjoint from dev/test (A1 saw all of train).
    'train'  : the first n of load_train(n, seed) (same as A2)."""
    if kind == "train":
        rows = load_train(n, seed=seed)
    else:
        from data import _load

        used = set(SPLIT_IDS["test"]) | set(SPLIT_IDS["dev"])
        ds = _load("validation")
        pool = [i for i, r in enumerate(ds["id"]) if r not in used]
        rng = np.random.default_rng(seed + 11)
        rows = [dict(ds[int(i)]) for i in rng.choice(pool, n, replace=False)]
    out = []
    for r in rows:
        titles, paras = paragraphs_of(r)
        out.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                    "type": r["type"], "paragraphs": paras, "gold": gold_mask(r)})
    return out


# ===================================================================== forward server
class Forward:
    def __init__(self, run: Path, model_dir: Path, track: str, budget: int, practice: list[dict]):
        from transformers import AutoTokenizer

        self.run, self.model_dir, self.track = run, model_dir, track
        self.budget, self.used = budget, 0
        self.practice = practice
        self.base = el.load_base(model_dir)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.orig = el.build_model(model_dir, el.EditState(track), self.base)
        self.cache_key, self.cache_model = None, None
        self.lock = threading.Lock()
        self.log = (run / "forward_log.jsonl").open("a", encoding="utf-8")
        (run / "fwd").mkdir(exist_ok=True)

    def edited_model(self):
        p = self.run / "work" / "edited" / "state.json"
        key = p.read_text() if p.exists() else ""
        raw = self.run / "work" / "edited" / "raw.safetensors"
        key += str(raw.stat().st_mtime) if raw.exists() else ""
        if key != self.cache_key:
            st = el.EditState.load(self.run / "work" / "edited", self.track)
            self.cache_model = el.build_model(self.model_dir, st, self.base)
            self.cache_key = key
        return self.cache_model

    @torch.no_grad()
    def run_model(self, model, q, p, cls, mlp, attn_n):
        out = {"scores": [], "cls": [], "mlp": [], "attn": []}
        hooks, acts = [], []
        if mlp:
            for layer in model.bert.encoder.layer:
                hooks.append(layer.intermediate.register_forward_hook(
                    lambda m, i, o: acts.append(o[:, 0].float().numpy())))
        try:
            for s in range(0, len(q), 32):
                enc = self.tok(q[s:s + 32], p[s:s + 32], truncation=True, max_length=256,
                               padding=True, return_tensors="pt")
                acts.clear()
                want_attn = attn_n > 0 and s == 0
                o = model(**enc, output_hidden_states=cls, output_attentions=want_attn)
                out["scores"].append(o.logits.squeeze(-1).numpy())
                if cls:
                    out["cls"].append(np.stack([h[:, 0].numpy() for h in o.hidden_states], 1))
                if mlp:
                    out["mlp"].append(np.stack(acts, 1))
                if want_attn:
                    out["attn"] = [np.stack([a[k].numpy() for a in o.attentions]) for k in range(min(attn_n, len(q)))]
                    out["attn_tokens"] = [self.tok.convert_ids_to_tokens(enc["input_ids"][k]) for k in range(min(attn_n, len(q)))]
        finally:
            for h in hooks:
                h.remove()
        res = {"scores": np.concatenate(out["scores"])}
        if cls:
            res["cls"] = np.concatenate(out["cls"])
        if mlp:
            res["mlp"] = np.concatenate(out["mlp"])
        if attn_n:
            res["attn"] = np.array(out["attn"], dtype=object)
            res["attn_tokens"] = np.array(out.get("attn_tokens", []), dtype=object)
        return res

    def handle(self, req: dict) -> dict:
        with self.lock:
            if self.used >= self.budget:
                return {"ok": False, "error": "forward-pass budget exhausted"}
            items, pairs = req.get("items"), req.get("pairs")
            if (items is None) == (pairs is None):
                return {"ok": False, "error": "give exactly one of items or pairs"}
            if items is not None:
                if len(items) > LIMIT_ITEMS:
                    return {"ok": False, "error": f"max {LIMIT_ITEMS} items"}
                try:
                    q = [self.practice[int(a)]["question"] for a, _ in items]
                    p = [self.practice[int(a)]["paragraphs"][int(b)] for a, b in items]
                except (IndexError, ValueError, TypeError) as e:
                    return {"ok": False, "error": f"bad items: {e}"}
            else:
                if len(pairs) > LIMIT_PAIRS:
                    return {"ok": False, "error": f"max {LIMIT_PAIRS} pairs"}
                q, p = [str(a) for a, _ in pairs], [str(b) for _, b in pairs]
            which = req.get("model", "edited")
            if which not in ("edited", "original", "both"):
                return {"ok": False, "error": "model must be edited|original|both"}
            attn_n = min(int(req.get("attn_n") or 0), LIMIT_ATTN)
            try:
                res = {}
                for name in (["edited", "original"] if which == "both" else [which]):
                    m = self.edited_model() if name == "edited" else self.orig
                    r = self.run_model(m, q, p, bool(req.get("cls")), bool(req.get("mlp")), attn_n)
                    res.update({f"{k}_{name}": v for k, v in r.items()})
            except Exception as e:  # noqa: BLE001 - failed passes are not counted
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self.used += 1
            path = self.run / "fwd" / f"pass_{self.used:03d}.npz"
            np.savez(path, **res)
            self.log.write(json.dumps({"pass": self.used, "t": time.time(), "model": which,
                                       "n": len(q), "kind": "items" if items is not None else "pairs",
                                       "cls": bool(req.get("cls")), "mlp": bool(req.get("mlp")),
                                       "attn_n": attn_n}) + "\n")
            self.log.flush()
            return {"ok": True, "npz": str(path), "passes_left": self.budget - self.used}


def serve(fwd: Forward) -> int:
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers["Content-Length"])
            body = json.dumps(fwd.handle(json.loads(self.rfile.read(n)))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


# ===================================================================== kernel process
class Kernel:
    def __init__(self, run: Path, env: dict):
        self.rundir, self.env = run, env
        self.start()

    def start(self):
        self.p = subprocess.Popen([sys.executable, str(HERE / "kernel.py")], cwd=str(HERE),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", env=self.env, bufsize=1)
        self._read(120)

    def _read(self, timeout):
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("line", self.p.stdout.readline()), daemon=True)
        t.start()
        t.join(timeout)
        if "line" not in box:
            return None
        line = box["line"]
        try:
            return json.loads(line)["output"]
        except (json.JSONDecodeError, KeyError):
            return line

    def run(self, code: str) -> str:
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.flush()
        out = self._read(EXEC_TIMEOUT)
        if out is None:
            self.p.kill()
            self.start()
            return (f"[harness] execution exceeded {EXEC_TIMEOUT}s; the kernel was restarted. "
                    "Variables are lost; saved edits and files in /work are kept.")
        return out


# ===================================================================== editors
class AstraEditor:
    def __init__(self, run: Path, model: str, effort: str):
        self.run, self.model, self.effort = run, model, effort
        self.thread = None
        self.cwd = run / "codex_cwd"
        self.cwd.mkdir(exist_ok=True)

    def new_thread(self):
        self.thread = None

    def send(self, msg: str, turn_dir: Path) -> tuple[str, dict]:
        base = [CODEX_BIN, "exec"] + (["resume", self.thread] if self.thread else [])
        cmd = base + ["-m", self.model, *CODEX_OFF, *CODEX_CFG,
                      "-c", f"model_reasoning_effort={self.effort}",
                      "--skip-git-repo-check", "--json", "-"]
        if not self.thread:
            cmd[2:2] = ["-s", "read-only"]
        t0 = time.time()
        p = subprocess.run(cmd, input=msg, capture_output=True, text=True, encoding="utf-8",
                           cwd=str(self.cwd))
        (turn_dir / "events.jsonl").write_text(p.stdout, encoding="utf-8")
        (turn_dir / "stderr.txt").write_text(p.stderr, encoding="utf-8")
        texts, usage, bad = [], {}, []
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "thread.started" and not self.thread:
                self.thread = ev["thread_id"]
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage", {})
            item = ev.get("item") or {}
            if ev.get("type") == "item.completed":
                if item.get("type") == "agent_message":
                    texts.append(item.get("text", ""))
                elif item.get("type") not in ALLOWED_ITEMS:
                    bad.append(item.get("type"))
        meta = {"usage": usage, "seconds": round(time.time() - t0, 1), "tool_items": bad,
                "returncode": p.returncode, "thread": self.thread}
        return "\n\n".join(texts), meta


class DummyEditor:
    def __init__(self, script: Path):
        self.replies = json.loads(Path(script).read_text(encoding="utf-8"))
        self.i = 0

    def new_thread(self):
        pass

    def send(self, msg: str, turn_dir: Path) -> tuple[str, dict]:
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r, {"usage": {}, "seconds": 0, "tool_items": [], "returncode": 0, "thread": "dummy"}


# ===================================================================== main loop
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def audit_code(code: str) -> tuple[list[str], list[str]]:
    hard = [p for p in HARD if re.search(p, code)]
    soft = [p for p in SOFT if re.search(p, code)]
    return hard, soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--track", choices=["C", "R"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--turn-cap", type=int, required=True)
    ap.add_argument("--editor", choices=["astra", "dummy"], required=True)
    ap.add_argument("--model-name", default="gpt-6-astra")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--script", help="dummy editor replies (JSON list)")
    ap.add_argument("--model-dir", default=str(HERE / "models" / "a1"))
    ap.add_argument("--practice", choices=["heldval", "train"], default="heldval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt", default=str(HERE / "prompt_trackC.md"))
    args = ap.parse_args()

    run = HERE / "runs" / args.run
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} exists; choose a new --run name")
    (run / "turns").mkdir(parents=True)
    (run / "work" / "edited").mkdir(parents=True)
    (run / "work" / "notes.md").write_text("", encoding="utf-8")
    el.EditState(args.track).save(run / "work" / "edited")

    practice = build_practice(args.practice, args.n)
    (run / "practice.json").write_text(json.dumps(practice), encoding="utf-8")
    fwd = Forward(run, Path(args.model_dir), args.track, args.budget, practice)
    port = serve(fwd)
    env = dict(os.environ, RUN_DIR=str(run), MODEL_DIR=str(args.model_dir), TRACK=args.track,
               FWD_URL=f"http://127.0.0.1:{port}/", PYTHONIOENCODING="utf-8")
    kernel = Kernel(run, env)

    base_cost = el.EditState(args.track).cost()
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    for k, v in {"N": args.n, "B": args.budget, "TURN_CAP": args.turn_cap,
                 "GFLOPS": round(base_cost["encoder_gflops_per_question"], 2),
                 "PARAMS": base_cost["encoder_params"]}.items():
        prompt = prompt.replace("{" + k + "}", str(v))
    (run / "prompt.md").write_text(prompt, encoding="utf-8")
    json.dump(vars(args) | {"started": time.strftime("%Y-%m-%d %H:%M:%S")},
              (run / "config.json").open("w"), indent=1)

    editor = (AstraEditor(run, args.model_name, args.effort) if args.editor == "astra"
              else DummyEditor(args.script))
    msg, records, after_budget, since_handoff = prompt, [], 0, 0
    for turn in range(1, args.turn_cap + 1):
        td = run / "turns" / f"t{turn:03d}"
        td.mkdir()
        (td / "msg_in.md").write_text(msg, encoding="utf-8")
        reply, meta = editor.send(msg, td)
        (td / "reply.md").write_text(reply, encoding="utf-8")
        m = CODE_RE.search(reply)
        code = m.group(1) if m else None
        rec = {"turn": turn, **meta, "has_code": bool(code), "passes_used": fwd.used}
        if code:
            (td / "code.py").write_text(code, encoding="utf-8")
            hard, soft = audit_code(code)
            rec |= {"hard_flags": hard, "soft_flags": soft}
            if hard:
                out = ("[harness] code refused before execution; it matches forbidden patterns "
                       f"{hard}. See RULES.")
            else:
                out = kernel.run(code)
        else:
            out = "[harness] no ```python block found in your reply."
        if len(out) > OUT_CAP:
            out = out[:OUT_CAP] + f"\n[harness] output cut at {OUT_CAP} characters"
        (td / "output.txt").write_text(out, encoding="utf-8")
        records.append(rec)
        (run / "run.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
        left = fwd.budget - fwd.used
        print(f"turn {turn}: code={bool(code)} passes_left={left} tools={meta['tool_items']} "
              f"{meta['seconds']}s", flush=True)

        if left == 0:
            after_budget += 1
            if not code and after_budget > 1:
                break  # final reply after the budget was spent
            if after_budget > 3:
                break
        status = (f"Passes left: {left}/{args.budget}. Turn {turn}/{args.turn_cap}.")
        if left == 0:
            status += (" The forward-pass budget is used up. Code still runs (forward() is refused). "
                       "Make sure /work/edited holds your final edit, then reply WITHOUT a code block "
                       "with your final explanation (mechanism, changes, evidence).")
        msg = f"OUTPUT:\n```\n{out}\n```\n{status}"
        since_handoff += 1
        if since_handoff >= HANDOFF_EVERY:
            editor.new_thread()
            since_handoff = 0
            notes = (run / "work" / "notes.md").read_text(encoding="utf-8")[-20000:]
            msg = (prompt + "\n\n---\nHANDOFF: this continues an earlier session. Your notes "
                   f"(/work/notes.md):\n{notes}\n\nCurrent edit: kernel `edited()`; {status}\n"
                   f"Last output:\n```\n{out[:3000]}\n```")
    kernel.p.kill()
    final = el.EditState.load(run / "work" / "edited", args.track)
    summary = {"turns": len(records), "passes_used": fwd.used,
               "tool_violations": sum(len(r["tool_items"]) for r in records),
               "hard_flags": sum(len(r.get("hard_flags", [])) for r in records),
               "soft_flag_turns": sum(bool(r.get("soft_flags")) for r in records),
               "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in records),
               "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in records),
               "editor_seconds": sum(r["seconds"] for r in records),
               "final_cost": final.cost(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
