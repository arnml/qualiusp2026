# Hypothesis: task-irrelevant components have little activation energy, little ranking signal, or mostly structural-token attention.
def question_metrics(items, scores):
    grouped = {}
    for (q, p), score in zip(items, np.asarray(scores).reshape(-1)):
        grouped.setdefault(q, []).append((p, float(score)))
    result = {}
    for q, entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        ps = np.array([p for p, s in entries])
        ss = np.array([s for p, s in entries])
        gg = np.array([bool(practice[q]["gold"][p]) for p in ps])
        top = np.argsort(-ss, kind="stable")[:3]
        negatives = np.sort(ss[~gg])[::-1]
        margin = float(ss[gg].min() - negatives[1]) if len(negatives) >= 2 else float("inf")
        result[q] = {"success": bool(gg[top].sum() == gg.sum()), "margin": margin}
    return result

baseline_metrics = question_metrics(baseline_items, baseline["scores_original"])
print("Selection baseline both-gold@3:", sum(v["success"] for v in baseline_metrics.values()), "/", len(baseline_metrics))
print("Selection margin quantiles:", quantiles([v["margin"] for v in baseline_metrics.values()]))

labels = np.array([practice[q]["gold"][p] for q, p in baseline_items], dtype=np.float32)
centered_labels = labels.copy()
centered_scores = baseline["scores_original"].copy()
for q in selection_q:
    ix = np.array([i for i, item in enumerate(baseline_items) if item[0] == q])
    centered_labels[ix] -= centered_labels[ix].mean()
    centered_scores[ix] -= centered_scores[ix].mean()

readout_direction = (
    load_original("bert.pooler.dense.weight").T @ load_original("classifier.weight").flatten()
).numpy()
readout_direction /= np.linalg.norm(readout_direction)
cls_states = baseline["cls_original"]
mlp_importance = {}
for layer in range(6):
    a = baseline["mlp_original"][:, layer, :]
    rms = np.sqrt(np.mean(a.astype(np.float64)**2, axis=0))
    importance = rms * weight_stats[layer]["mlp_output_norm"]
    mlp_importance[layer] = importance
    ranked = np.argsort(importance)
    energy = importance**2
    task_cov = np.abs(centered_labels @ a / len(a))
    prev, curr = cls_states[:, layer, :], cls_states[:, layer+1, :]
    delta = curr - prev
    cosine = np.sum(prev*curr, axis=1) / (np.linalg.norm(prev, axis=1)*np.linalg.norm(curr, axis=1))
    projection = delta @ readout_direction
    print("Layer", layer,
          "CLS cosine", round(float(cosine.mean()), 4),
          "relative delta", round(float(np.mean(np.linalg.norm(delta, axis=1)/np.linalg.norm(prev, axis=1))), 4),
          "delta/readout score correlation", round(float(np.corrcoef(projection, centered_scores)[0,1]), 4))
    print("  MLP importance quantiles:", quantiles(importance),
          "bottom 25/50% energy fractions:",
          [round(float(energy[ranked[:n]].sum()/energy.sum()), 4) for n in (384, 768)],
          "activation RMS quantiles:", quantiles(rms))

attention_summary = {}
for layer in range(6):
    records = []
    for ex in range(8):
        tokens = [str(t) for t in baseline["attn_tokens_original"][ex]]
        valid = np.array([t != "[PAD]" for t in tokens])
        structure = np.array([t in ("[CLS]", "[SEP]", ".", ",", ":", ";", "!", "?", "-", "(", ")", '"', "'") for t in tokens]) & valid
        seps = [i for i, t in enumerate(tokens) if t == "[SEP]"]
        question_mask = np.zeros(len(tokens), dtype=bool)
        paragraph_mask = np.zeros(len(tokens), dtype=bool)
        question_mask[1:seps[0]] = True
        paragraph_mask[seps[0]+1:seps[-1]] = True
        a = np.asarray(baseline["attn_original"][ex, layer], dtype=np.float32)
        cls_struct = a[:, 0, :][:, structure].sum(axis=1)
        all_struct = a[:, valid, :][:, :, structure].sum(axis=2).mean(axis=1)
        question_to_para = a[:, question_mask, :][:, :, paragraph_mask].sum(axis=2).mean(axis=1)
        entropy = -(a[:, 0, :] * np.log(np.maximum(a[:, 0, :], 1e-12))).sum(axis=1)
        records.append(np.stack([cls_struct, all_struct, question_to_para, entropy], axis=1))
    attention_summary[layer] = np.mean(records, axis=0)
    print("Layer", layer, "head: CLS-struct / all-struct / question-to-paragraph / CLS-entropy")
    print([(h, *np.round(attention_summary[layer][h], 3).tolist()) for h in range(12)])

with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nSelection baseline successes: " + str(sum(v["success"] for v in baseline_metrics.values())) + "/100.\n")
    f.write("Statistics: activation RMS times MLP output-column norm; CLS interlayer cosine and relative change; "
            "correlation of CLS changes projected on the static pooler/classifier weight direction with scores; "
            "structural-token attention and question-to-paragraph attention. No fitted objectives.\n")
