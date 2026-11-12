"""Functions available inside the editor's kernel. Imported into the kernel namespace.

Environment variables (set by the harness): RUN_DIR, MODEL_DIR, TRACK, FWD_URL.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import editlib as _el

RUN_DIR = Path(os.environ["RUN_DIR"])
MODEL_DIR = Path(os.environ["MODEL_DIR"])
TRACK = os.environ["TRACK"]
_FWD = os.environ["FWD_URL"]
WORK = RUN_DIR / "work"
EDIT_DIR = WORK / "edited"
NOTES = WORK / "notes.md"
WORK.mkdir(parents=True, exist_ok=True)

_BASE = _el.load_base(MODEL_DIR)
NAMES = sorted(_BASE)
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
practice = json.loads((RUN_DIR / "practice.json").read_text(encoding="utf-8"))


def _state() -> _el.EditState:
    return _el.EditState.load(EDIT_DIR, TRACK)


def _commit(s: _el.EditState):
    errs = s.check(_BASE)
    if errs:
        raise ValueError("edit refused: " + "; ".join(errs))
    s.save(EDIT_DIR)


def load(name: str) -> torch.Tensor:
    """Current tensor (edit applied if any), float32."""
    return _state().apply(_BASE)[name].clone()


def load_original(name: str) -> torch.Tensor:
    return _BASE[name].clone()


def edited() -> dict:
    s = _state()
    out = {"track": s.track, "heads_removed": sorted(s.heads), "neurons_removed": len(s.neurons),
           "layers_removed": sorted(s.layers), "scalings": len(s.scales), "cost": s.cost()}
    if s.raw:
        out["raw_changed_entries"] = {k: int((v != _BASE[k]).sum()) for k, v in s.raw.items()}
    return out


def cost() -> dict:
    """Approximate encoder FLOPs per question (10 paragraphs) and parameters."""
    return _state().cost()


def reset_all():
    """Discard every edit (back to the original model)."""
    _el.EditState(TRACK).save(EDIT_DIR)


# ---------------------------------------------------------------- Track C: removals
def remove_head(layer: int, head: int):
    s = _state(); s.heads.add((int(layer), int(head))); _commit(s)


def restore_head(layer: int, head: int):
    s = _state(); s.heads.discard((int(layer), int(head))); _commit(s)


def remove_neurons(layer: int, idx):
    s = _state(); s.neurons |= {(int(layer), int(i)) for i in np.atleast_1d(idx)}; _commit(s)


def restore_neurons(layer: int, idx):
    s = _state(); s.neurons -= {(int(layer), int(i)) for i in np.atleast_1d(idx)}; _commit(s)


def remove_layer(layer: int):
    s = _state(); s.layers.add(int(layer)); _commit(s)


def restore_layer(layer: int):
    s = _state(); s.layers.discard(int(layer)); _commit(s)


def scale_rows(name: str, rows, f: float):
    """Multiply rows of a kept weight/bias (output dimension) by f, e.g. to compensate a removal."""
    s = _state(); s.scales.append({"name": name, "axis": "rows", "idx": [int(i) for i in np.atleast_1d(rows)], "f": float(f)}); _commit(s)


def scale_cols(name: str, cols, f: float):
    """Multiply columns of a kept weight (input dimension) by f."""
    s = _state(); s.scales.append({"name": name, "axis": "cols", "idx": [int(i) for i in np.atleast_1d(cols)], "f": float(f)}); _commit(s)


def drop_scalings():
    s = _state(); s.scales = []; _commit(s)


# ---------------------------------------------------------------- Track R: raw edits
def save_edit(name: str, tensor: torch.Tensor):
    if TRACK != "R":
        raise ValueError("save_edit is only available in Track R; use the removal functions")
    s = _state(); s.raw[name] = tensor.detach().float().clone(); _commit(s)


def drop_edit(name: str):
    s = _state(); s.raw.pop(name, None); _commit(s)


# ---------------------------------------------------------------- forward passes (budgeted)
def forward(items=None, pairs=None, model: str = "edited", cls: bool = False,
            mlp: bool = False, attn_n: int = 0) -> dict:
    """Run the network. ONE call = ONE forward pass of the budget (any size up to the limits).

    items : list of (question_index, paragraph_index) into `practice` (<= 2048)
    pairs : list of (question_text, paragraph_text) you write yourself (<= 64)
    model : "edited" | "original" | "both"
    cls   : also return [CLS] hidden state after embeddings and after every kept layer
    mlp   : also return MLP activations (after GELU) at [CLS] for every kept layer
    attn_n: also return attention probabilities for the first attn_n inputs (<= 8)
    Returns numpy arrays: scores_<model>, and cls_<model>, mlp_<model>, attn_<model>, tokens.
    """
    req = {"items": items, "pairs": pairs, "model": model, "cls": cls, "mlp": mlp, "attn_n": attn_n}
    r = urllib.request.Request(_FWD, data=json.dumps(req, default=int).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        out = json.loads(resp.read())
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "forward failed"))
    data = dict(np.load(out["npz"], allow_pickle=True))
    data["passes_left"] = out["passes_left"]
    print(f"[forward] pass used; {out['passes_left']} left")
    return data
