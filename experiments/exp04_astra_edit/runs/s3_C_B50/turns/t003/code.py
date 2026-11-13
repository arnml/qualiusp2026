# Hypothesis: channels with small activation-weight products carry little CLS signal, while separator-focused heads may be redundant.
base_s = np.asarray(diag["scores_original"]).reshape(-1)

def question_metrics(scores, items):
    scores = np.asarray(scores).reshape(-1)
    result = {}
    for q in sorted(set(q for q, _ in items)):
        positions = [i for i, (qi, _) in enumerate(items) if qi == q]
        ps = np.array([items[i][1] for i in positions])
        ss = scores[positions]
        gg = np.asarray(practice[q]["gold"])[ps].astype(bool)
        top = np.argsort(-ss, kind="stable")[:3]
        ok = bool(gg[top].sum() == gg.sum())
        neg = np.sort(ss[~gg])[::-1]
        margin = float(ss[gg].min() - neg[1]) if len(neg) >= 2 else float("inf")
        result[q] = (ok, margin)
    return result

base_m = question_metrics(base_s, select_items)
history = []

def report(label, output, items=select_items, baseline=None):
    ss = np.asarray(output["scores_edited"]).reshape(-1)
    original = np.asarray(output["scores_original"]).reshape(-1) if "scores_original" in output else baseline
    if original is None:
        original = base_s
    bm = question_metrics(original, items)
    em = question_metrics(ss, items)
    lost = [q for q in bm if bm[q][0] and not em[q][0]]
    gained = [q for q in bm if not bm[q][0] and em[q][0]]
    dm = np.array([em[q][1] - bm[q][1] for q in bm])
    row = dict(label=label, n=len(bm), original=sum(x[0] for x in bm.values()),
               edited=sum(x[0] for x in em.values()), lost=lost, gained=gained,
               margin_delta_quantiles=np.round(np.quantile(dm, [.1, .5, .9]), 4).tolist(),
               score_correlation=float(np.corrcoef(original, ss)[0, 1]),
               cost=cost())
    history.append(row)
    print(row)
    with NOTES.open("a") as f:
        f.write("\n" + repr(row) + "\n")
    return row

print("Selection baseline:", sum(v[0] for v in base_m.values()), "/", len(base_m))
cl = diag["cls_original"]
act = diag["mlp_original"]
readout = arr(load_original("classifier.weight")) @ arr(load_original("bert.pooler.dense.weight"))
readout = readout.reshape(-1)
readout /= np.linalg.norm(readout)
neuron_ranks = {}
neuron_importance = {}
for l in range(6):
    wo = arr(load_original(wn(l, "output.dense.weight")))
    rms = np.sqrt(np.mean(act[:, l, :] ** 2, axis=0))
    importance = rms * np.linalg.norm(wo, axis=0)
    neuron_importance[l] = importance
    neuron_ranks[l] = np.argsort(importance)
    delta = cl[:, l + 1, :] - cl[:, l, :]
    print("Layer", l,
          "CLS cosine", round(float(np.mean(np.sum(cl[:, l] * cl[:, l+1], axis=1) /
                                   (np.linalg.norm(cl[:, l], axis=1) * np.linalg.norm(cl[:, l+1], axis=1)))), 4),
          "delta RMS", round(float(np.sqrt(np.mean(delta**2))), 4),
          "readout delta RMS", round(float(np.sqrt(np.mean((delta @ readout)**2))), 4),
          "neuron importance q", np.round(np.quantile(importance, [0, .1, .25, .5, .9, 1]), 4).tolist())
print("Attention token example:", diag["attn_tokens_original"][0, :30].tolist())
print("Tokenizer special IDs:", tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
