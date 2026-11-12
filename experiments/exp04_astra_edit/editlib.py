"""Edit state for the 22.7M BERT cross-encoder: removals (Track C) and raw edits (Track R).

The edit state is a small JSON + tensor file in <run>/work/edited/ so that the kernel
(which writes it) and the forward server (which reads it) never share Python objects.

Track C ("compress") only allows structured removal plus row/column rescaling:
  remove_head(l, h)    -> head h of layer l is deleted (Q/K/V rows, O columns)
  remove_neuron(l, i)  -> MLP neuron i of layer l is deleted (in row, out column)
  remove_layer(l)      -> layer l is skipped entirely
  scale_rows(name, rows, f) / scale_cols(name, cols, f) -> multiply kept rows/cols by f
Track R ("repair") allows raw tensor edits in scope with a cap on changed entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

N_LAYERS, HIDDEN, N_HEADS, HEAD, FFN = 6, 384, 12, 32, 1536
SCOPE_R = ("bert.encoder.layer.", "bert.pooler.dense.")
RAW_CAP = 108_000  # Track R: max changed entries vs original

PFX = "bert.encoder.layer.{l}."
Q, K, V = (PFX + "attention.self.query", PFX + "attention.self.key", PFX + "attention.self.value")
O = PFX + "attention.output.dense"
FI, FO = PFX + "intermediate.dense", PFX + "output.dense"


def load_base(model_dir: Path) -> dict[str, torch.Tensor]:
    sd = load_file(str(Path(model_dir) / "model.safetensors"))
    return {k: v.float() for k, v in sd.items()}


class EditState:
    """Serializable edit description. Applied on top of the base state dict."""

    def __init__(self, track: str):
        assert track in ("C", "R")
        self.track = track
        self.heads: set[tuple[int, int]] = set()
        self.neurons: set[tuple[int, int]] = set()
        self.layers: set[int] = set()
        self.scales: list[dict] = []          # {"name","axis","idx","f"}
        self.raw: dict[str, torch.Tensor] = {}  # Track R only

    # ---------------------------------------------------------------- persistence
    def save(self, d: Path):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"track": self.track, "heads": sorted(self.heads), "neurons": sorted(self.neurons),
                "layers": sorted(self.layers), "scales": self.scales, "raw": sorted(self.raw)}
        (d / "state.json").write_text(json.dumps(meta))
        if self.raw:
            save_file({k: v.contiguous() for k, v in self.raw.items()}, str(d / "raw.safetensors"))
        elif (d / "raw.safetensors").exists():
            (d / "raw.safetensors").unlink()

    @classmethod
    def load(cls, d: Path, track: str = "C") -> EditState:
        d = Path(d)
        if not (d / "state.json").exists():
            return cls(track)
        m = json.loads((d / "state.json").read_text())
        s = cls(m["track"])
        s.heads = {tuple(x) for x in m["heads"]}
        s.neurons = {tuple(x) for x in m["neurons"]}
        s.layers = set(m["layers"])
        s.scales = m["scales"]
        if m["raw"]:
            s.raw = load_file(str(d / "raw.safetensors"))
        return s

    # ---------------------------------------------------------------- apply
    def apply(self, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sd = {k: v.clone() for k, v in base.items()}
        for name, t in self.raw.items():
            sd[name] = t.clone()
        for l, h in self.heads:
            sl = slice(h * HEAD, (h + 1) * HEAD)
            for P in (Q, K, V):
                sd[P.format(l=l) + ".weight"][sl, :] = 0
                sd[P.format(l=l) + ".bias"][sl] = 0
            sd[O.format(l=l) + ".weight"][:, sl] = 0
        for l, i in self.neurons:
            sd[FI.format(l=l) + ".weight"][i, :] = 0
            sd[FI.format(l=l) + ".bias"][i] = 0
            sd[FO.format(l=l) + ".weight"][:, i] = 0
        for s in self.scales:
            w = sd[s["name"]]
            idx = torch.tensor(s["idx"], dtype=torch.long)
            if s["axis"] == "rows":
                w[idx] *= s["f"]
            else:
                w[:, idx] *= s["f"]
        return sd

    # ---------------------------------------------------------------- checks
    def check(self, base: dict[str, torch.Tensor]) -> list[str]:
        errs = []
        for l, h in self.heads:
            if not (0 <= l < N_LAYERS and 0 <= h < N_HEADS):
                errs.append(f"bad head {(l, h)}")
        for l, i in self.neurons:
            if not (0 <= l < N_LAYERS and 0 <= i < FFN):
                errs.append(f"bad neuron {(l, i)}")
        for l in self.layers:
            if not 0 <= l < N_LAYERS:
                errs.append(f"bad layer {l}")
        if len(self.layers) >= N_LAYERS:
            errs.append("cannot remove all layers")
        if self.track == "C" and self.raw:
            errs.append("Track C allows only removals and row/column scaling")
        for s in self.scales:
            if not s["name"].startswith(SCOPE_R) or s["name"].endswith("LayerNorm.weight"):
                errs.append(f"scaling out of scope: {s['name']}")
        if self.track == "R":
            changed = 0
            for name, t in self.raw.items():
                if not name.startswith(SCOPE_R):
                    errs.append(f"out of scope: {name}")
                    continue
                if t.shape != base[name].shape:
                    errs.append(f"shape mismatch: {name}")
                    continue
                diff = t != base[name]
                nz = base[name] != 0
                if diff[nz].all() and nz.any():
                    ratio = t[nz] / base[name][nz]
                    if float(ratio.std()) < 1e-6:
                        errs.append(f"whole-tensor rescale refused: {name}")
                changed += int(diff.sum())
            if changed > RAW_CAP:
                errs.append(f"{changed} changed entries > cap {RAW_CAP}")
        return errs

    # ---------------------------------------------------------------- cost
    def cost(self, seq_len: float = 170.0, pairs: int = 10) -> dict:
        """Approximate inference FLOPs per question (pairs x seq_len tokens) and params."""
        T = seq_len
        fl, params = 0.0, 0
        for l in range(N_LAYERS):
            if l in self.layers:
                continue
            h = N_HEADS - sum(1 for (a, _) in self.heads if a == l)
            n = FFN - sum(1 for (a, _) in self.neurons if a == l)
            qkvo = 4 * HIDDEN * (h * HEAD)
            mlp = 2 * HIDDEN * n
            params += qkvo + mlp + 3 * h * HEAD + HIDDEN + n + HIDDEN + 4 * HIDDEN  # biases + LNs
            fl += 2 * T * (qkvo + mlp) + 2 * 2 * T * T * h * HEAD  # matmuls + scores/context
        base_fl = 2 * T * HIDDEN * HIDDEN  # pooler on CLS is negligible; embeddings are lookups
        return {"encoder_gflops_per_question": pairs * (fl + base_fl) / 1e9,
                "encoder_params": params,
                "removed": {"heads": len(self.heads), "neurons": len(self.neurons),
                            "layers": sorted(self.layers)}}


def build_model(model_dir: Path, state: EditState, base: dict | None = None):
    """HF model with the edit applied; removed layers are physically dropped."""
    from transformers import AutoModelForSequenceClassification

    base = base if base is not None else load_base(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), attn_implementation="eager")
    model.load_state_dict(state.apply(base))
    if state.layers:
        keep = [m for i, m in enumerate(model.bert.encoder.layer) if i not in state.layers]
        model.bert.encoder.layer = torch.nn.ModuleList(keep)
        model.config.num_hidden_layers = len(keep)
    return model.eval()
