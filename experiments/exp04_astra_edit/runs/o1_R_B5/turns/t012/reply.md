Two results narrow things down:
- **Cross-question calibration is not the bottleneck.** Oracle per-question centering does not help (A gets worse, B better), so within-question ranking is what matters.
- **The leftover gold information is mostly paragraph length at layers 3–4** (corr with token count −0.77 and −0.65), a confound I won't exploit. At layer 5 output (C5) it is instead tied to title-match (corr 0.32) and separates gold on B by 0.59 sd. By the final layer that information has shrunk (0.41 sd; hard > beating only 48%).

Next I'll test which feature re-weighting would actually help, on split A as a diagnostic only, never written into weights. The contrasts are hard gold vs beating distractors: title match 0.46 vs 0.24, body overlap 0.41 vs 0.44, answer present 0.71 vs 0.10. I'll also inspect attention for the 8 recorded paragraphs of question 0 to see whether any head reads the title span.

```python
# Hypothesis: re-weighting toward title-match (paragraph is ABOUT a question entity) and away from body overlap would help; some head at [CLS] reads the title span.
z = lambda v: (v - v.mean()) / cq(v).std()
body_ov = []
for qi_, q in enumerate(practice):
    qw = content(q["question"])
    for p in q["paragraphs"]:
        body = p.split(". ", 1)[1] if ". " in p else p; body_ov.append(len(qw & content(body)) / max(1, len(qw)))
F["body_ov"] = body_ov
rows = []
for nm, feat, sign in [("+title_in_q", F.title_in_q.values, 1), ("-overlap", F.overlap.values, -1), ("-body_ov", F.body_ov.values, -1), ("-ntok", F.ntok.values.astype(float), -1)]:
    for beta in [0.25, 0.5, 1.0]:
        s = S0 + sign * beta * sdS * z(feat); rows.append((nm, beta) + tuple(robust(FA, s[A]).values()))
print(pd.DataFrame(rows, columns=["feat", "beta"] + list(robust(FA, S0[A]).keys())).to_string(index=False))
# attention of [CLS] (row 0) over segments for q0's first 8 paragraphs: question span, title span, body span
att = np.asarray(out1["attn_original"]); toks = out1["attn_tokens_original"]
sep_pos = [[i for i, t in enumerate(tt) if t == "[SEP]"] for tt in toks]
res = np.zeros((6, 12, 3))
for k in range(8):
    tt = list(toks[k]); s1 = sep_pos[k][0]; dot = next((i for i in range(s1 + 1, len(tt)) if tt[i] == "."), s1 + 4)
    spans = [range(1, s1), range(s1 + 1, dot + 1), range(dot + 1, sep_pos[k][1] if len(sep_pos[k]) > 1 else len(tt))]
    for L in range(6):
        for h in range(12): res[L, h] += [att[k, L, h, 0, list(sp)].sum() / 8 for sp in spans]
for L in range(6): print("L%d CLS->title mass per head:" % L, np.round(res[L, :, 1], 2), "| ->question:", np.round(res[L, :, 0], 2))
```