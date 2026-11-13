Stdout is working; the first block's output had just been delayed. Next I'll build lexical features (question-word overlap, title-in-question, answer present) and spend pass 1 on a full diagnostic of the original model: all 1,994 paragraphs with [CLS] per layer, MLP activations, and attention for the first 8.

```python
# Hypothesis: gold paragraphs that lose are low-overlap "bridge" paragraphs; pass 1 = full diagnostic of the original model.
print(NOTES.read_text()[:800])
STOP = set("the a an of in on at to for by with from and or is was are were be been which what who whom whose when where how did does do that this as it its his her their they he she than also other".split())
def content(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1)
items, feats = [], []
for qi, q in enumerate(practice):
    qw = content(q["question"]); ans = q["answer"].lower().strip()
    for pi, p in enumerate(q["paragraphs"]):
        title = p.split(". ")[0]
        pw, tw = content(p), content(title)
        items.append((qi, pi))
        feats.append(dict(qi=qi, pi=pi, gold=q["gold"][pi], type=q["type"],
            overlap=len(qw & pw)/max(1, len(qw)), title_in_q=len(tw & qw)/max(1, len(tw)),
            has_ans=int(ans not in ("yes", "no") and ans in p.lower()),
            ntok=len(tok(p, add_special_tokens=False)["input_ids"])))
F = pd.DataFrame(feats); F["split"] = np.where(F.qi % 2 == 0, "A", "B")   # A = even q (choose), B = odd q (confirm)
print(F.groupby(["type", "gold"])[["overlap", "title_in_q", "has_ans", "ntok"]].mean().round(3))
out1 = forward(items=items, model="original", cls=True, mlp=True, attn_n=8)
print({k: (getattr(v, "shape", None) or v) for k, v in out1.items() if not isinstance(v, (list, dict))}, "| keys:", list(out1.keys()))
np.save(WORK/"scores_orig.npy", out1["scores_original"]); np.save(WORK/"cls_orig.npy", out1["cls_original"]); np.save(WORK/"mlp_orig.npy", out1["mlp_original"]); F.to_pickle(WORK/"feats.pkl")
```