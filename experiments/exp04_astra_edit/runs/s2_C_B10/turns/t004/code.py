# Hypothesis: the same component diagnostics remain valid after accounting for questions with fewer than two distractors.
def evaluate(scores, items):
    by_q = {}
    for i, (q, p) in enumerate(items):
        by_q.setdefault(q, []).append((p, float(scores[i])))
    result = {}
    for q, entries in by_q.items():
        ordered = sorted(entries, key=lambda x: x[1], reverse=True)
        gold = {p for p, g in enumerate(practice[q]["gold"]) if g}
        positive = [s for p, s in entries if p in gold]
        negative = sorted([s for p, s in entries if p not in gold], reverse=True)
        threshold_index = 3 - len(gold)
        threshold = negative[threshold_index] if len(negative) > threshold_index else -np.inf
        result[q] = {
            "ok": gold.issubset({p for p, _ in ordered[:3]}),
            "margin": min(positive) - threshold,
            "top3": tuple(p for p, _ in ordered[:3]),
        }
    return result

base_eval = evaluate(baseline["scores_original"], discovery_items)
finite_margins = [v["margin"] for v in base_eval.values() if np.isfinite(v["margin"])]
print("Discovery baseline:", sum(v["ok"] for v in base_eval.values()), "/", len(base_eval))
print("Baseline finite margin quantiles:", np.round(np.quantile(finite_margins, [0, .05, .1, .25, .5, 1]), 3))

labels = np.array([practice[q]["gold"][p] for q, p in discovery_items], dtype=bool)
cls_data = np.asarray(baseline["cls_original"], dtype=np.float32)
mlp_data = np.asarray(baseline["mlp_original"], dtype=np.float32)
neuron_stats, head_stats = {}, {}
weight_direction = (load_original("bert.pooler.dense.weight").T
                    @ load_original("classifier.weight").flatten()).numpy()
weight_direction /= np.linalg.norm(weight_direction)

for layer in range(6):
    prefix = f"bert.encoder.layer.{layer}."
    out = load_original(prefix + "output.dense.weight").numpy()
    inp = load_original(prefix + "intermediate.dense.weight").numpy()
    a = mlp_data[:, layer, :]
    output_norm = np.linalg.norm(out, axis=0)
    rms = np.sqrt(np.mean(a * a, axis=0))
    energy = rms * output_norm
    weight_size = np.linalg.norm(inp, axis=1) * output_norm
    contrast = np.abs(a[labels].mean(0) - a[~labels].mean(0)) * output_norm
    neuron_stats[layer] = dict(energy=energy, weight_size=weight_size, contrast=contrast,
                               rms=rms, output_norm=output_norm)
    before, after = cls_data[:, layer], cls_data[:, layer + 1]
    cosine = np.mean(np.sum(before * after, axis=1) /
                     (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)))
    direction_delta = (after - before) @ weight_direction
    print("Layer", layer, "CLS cosine", round(float(cosine), 3),
          "static readout contrast change", round(float(direction_delta[labels].mean() - direction_delta[~labels].mean()), 3),
          "MLP energy quantiles", np.round(np.quantile(energy, [0, .1, .25, .5, .9, 1]), 3),
          "near-inactive neurons", int(np.sum(rms < .01)))
    v = load_original(prefix + "attention.self.value.weight").numpy().reshape(12, 32, 384)
    o = load_original(prefix + "attention.output.dense.weight").numpy().reshape(384, 12, 32)
    strength = np.linalg.norm(v, axis=(1, 2)) * np.linalg.norm(o, axis=(0, 2))
    cls_structural, all_structural, cls_paragraph = np.zeros(12), np.zeros(12), np.zeros(12)
    for i in range(8):
        tokens = [str(t) for t in baseline["attn_tokens_original"][i]]
        valid = np.array([t != "[PAD]" for t in tokens])
        structural = np.array([t in {"[CLS]", "[SEP]", "[PAD]"} or
                               re.fullmatch(r"[^\w]+", t) is not None for t in tokens])
        sep = next(j for j, t in enumerate(tokens) if t == "[SEP]")
        paragraph = (np.arange(len(tokens)) > sep) & valid & ~structural
        attention = np.asarray(baseline["attn_original"][i, layer], dtype=np.float32)
        cls_structural += attention[:, 0, :][:, structural].sum(1) / 8
        all_structural += attention[:, valid, :][:, :, structural].sum(2).mean(1) / 8
        cls_paragraph += attention[:, 0, :][:, paragraph].sum(1) / 8
    head_stats[layer] = dict(strength=strength, cls_structural=cls_structural,
                             all_structural=all_structural, cls_paragraph=cls_paragraph)
    print(" head : weight strength / CLS structural / all structural / CLS paragraph")
    print(" ".join(f"{h}:{strength[h]:.1f}/{cls_structural[h]:.2f}/{all_structural[h]:.2f}/{cls_paragraph[h]:.2f}"
                   for h in range(12)))

with NOTES.open("a", encoding="utf-8") as f:
    f.write(f"\nDiscovery baseline both-gold@3: {sum(v['ok'] for v in base_eval.values())}/120.\n"
            "Diagnostics: MLP activation RMS times output-column norm; input/output weight norms; "
            "gold–distractor activation contrast; CLS layer cosine; head value/output norm product; "
            "structural-token and paragraph attention on eight probes. "
            "Pooler/classifier weight product is a static direction proxy, not a simulated forward or gradient.\n")
