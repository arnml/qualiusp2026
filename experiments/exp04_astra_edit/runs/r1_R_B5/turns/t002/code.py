# Hypothesis: failures concentrate on low-overlap gold paragraphs, while particular MLP neurons favor high-overlap distractors.
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def words(text):
    return set(w for w in re.findall(r"[a-z0-9]+", text.lower())
               if w not in ENGLISH_STOP_WORDS and len(w) > 1)

drows = np.array([item_row[x] for x in discovery_items])
dscores = baseline["scores_original"][drows]
dgold = np.array([practice[q]["gold"][p] for q, p in discovery_items], dtype=bool)
dlength = np.array([len(tok.encode(practice[q]["paragraphs"][p], add_special_tokens=False))
                    for q, p in discovery_items])
doverlap = np.array([
    len(words(practice[q]["question"]) & words(practice[q]["paragraphs"][p])) /
    max(1, len(words(practice[q]["question"]))) for q, p in discovery_items
])
dtitle = np.array([
    len(words(practice[q]["question"]) & words(practice[q]["paragraphs"][p].split(".")[0])) /
    max(1, len(words(practice[q]["question"]))) for q, p in discovery_items
])
dq = np.array([q for q, p in discovery_items])
groups_d = [np.flatnonzero(dq == q) for q in discovery_ids]

def centered(a, groups):
    a = np.array(a, dtype=np.float64, copy=True)
    for ix in groups:
        a[ix] -= a[ix].mean(axis=0)
    return a

def correlations(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean()
    return (x.T @ y) / np.maximum(1e-12, np.sqrt((x*x).sum(axis=0) * (y*y).sum()))

def metrics(scores, groups, gold, lengths):
    success, min_gold = [], []
    for ix in groups:
        ranking = ix[np.argsort(-scores[ix])]
        success.append(int(gold[ranking[:3]].sum() == gold[ix].sum()))
        min_gold.append(float(scores[ix][gold[ix]].min()))
    cutoff = float(np.sort(min_gold)[int(np.floor(0.1 * len(groups)))])
    costs, retained = [], []
    for ix in groups:
        keep = scores[ix] >= cutoff
        if not keep.any():
            keep[np.argmax(scores[ix])] = True
        costs.append(float(lengths[ix][keep].sum()))
        retained.append(bool(keep[gold[ix]].all()))
    return {"both@3": float(np.mean(success)), "tokens@90": float(np.mean(costs)),
            "retention": float(np.mean(retained)), "cutoff": cutoff,
            "success": np.array(success), "costs": np.array(costs),
            "min_gold": np.array(min_gold)}

base_d = metrics(dscores, groups_d, dgold, dlength)
print("Discovery baseline:", {k: v for k, v in base_d.items() if not isinstance(v, np.ndarray)})
print("Mean paragraph tokens:", round(float(dlength.mean()), 1))
print("Within-question score correlations:",
      {name: round(float(correlations(centered(x, groups_d), centered(dscores, groups_d))), 3)
       for name, x in [("word overlap", doverlap), ("title overlap", dtitle), ("length", dlength), ("gold", dgold)]})

weak_gold, competing_distractor = [], []
for ix in groups_d:
    gi, ni = ix[dgold[ix]], ix[~dgold[ix]]
    weak_gold.append(gi[np.argmin(dscores[gi])])
    competing_distractor.append(ni[np.argsort(-dscores[ni])[min(1, len(ni)-1)]])
weak_gold = np.array(weak_gold)
competing_distractor = np.array(competing_distractor)
print("Weak gold / second distractor means:",
      {"score": (round(float(dscores[weak_gold].mean()), 3), round(float(dscores[competing_distractor].mean()), 3)),
       "overlap": (round(float(doverlap[weak_gold].mean()), 3), round(float(doverlap[competing_distractor].mean()), 3)),
       "title_overlap": (round(float(dtitle[weak_gold].mean()), 3), round(float(dtitle[competing_distractor].mean()), 3))})

fail_order = np.argsort(dscores[weak_gold] - dscores[competing_distractor])
for j in fail_order[:7]:
    wg, nd = weak_gold[j], competing_distractor[j]
    qi, gp = discovery_items[wg]
    _, dp = discovery_items[nd]
    print("\nQ", qi, practice[qi]["type"], practice[qi]["question"])
    print("GOLD", round(float(dscores[wg]), 2), practice[qi]["paragraphs"][gp][:430])
    print("DISTRACTOR", round(float(dscores[nd]), 2), practice[qi]["paragraphs"][dp][:430])

neuron_stats = []
cs = centered(dscores, groups_d)
co = centered(doverlap, groups_d)
cg = centered(dgold, groups_d)
for layer in range(6):
    acts = baseline["mlp_original"][drows, layer, :]
    ca = centered(acts, groups_d)
    sd = np.maximum(ca.std(axis=0), 1e-6)
    rs, ro, rg = correlations(ca, cs), correlations(ca, co), correlations(ca, cg)
    contrast = (acts[competing_distractor].mean(axis=0) - acts[weak_gold].mean(axis=0)) / sd
    neuron_stats.append({"score": rs, "overlap": ro, "gold": rg, "contrast": contrast, "sd": sd})
    idx = np.argsort(-(np.maximum(rs, 0) * np.maximum(ro, 0) * np.maximum(contrast, 0)))[:6]
    print("Layer", layer, "candidate neurons (index, score r, overlap r, gold r, distractor contrast):",
          [(int(i), *[round(float(v[i]), 3) for v in [rs, ro, rg, contrast]]) for i in idx])

with NOTES.open("a") as f:
    f.write("\nBaseline discovery: " + str({k: v for k, v in base_d.items() if not isinstance(v, np.ndarray)}) + "\n")
