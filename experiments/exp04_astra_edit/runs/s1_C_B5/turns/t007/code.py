# Hypothesis: weak attention paths with little cross-segment communication can be removed on top of the validated MLP edit.
candidate_heads = [
    (0, 2),
    (1, 6), (1, 7), (1, 9),
    (2, 1), (2, 11),
    (3, 1),
    (4, 6), (4, 7), (4, 9),
    (5, 8), (5, 9),
]

print("Candidate A CLS drift by layer:")
for layer in range(6):
    orig = results["A_mlp"]["cls_original"][:, layer+1]
    edit = results["A_mlp"]["cls_edited"][:, layer+1]
    cosine = np.sum(orig*edit, axis=1) / (np.linalg.norm(orig, axis=1)*np.linalg.norm(edit, axis=1))
    print(layer, "mean cosine", round(float(cosine.mean()), 4))

a_orig = results["A_mlp"]["scores_original"].copy()
a_edit = results["A_mlp"]["scores_edited"].copy()
for q in selection_q:
    ix = np.array([i for i, item in enumerate(baseline_items) if item[0] == q])
    a_orig[ix] -= a_orig[ix].mean()
    a_edit[ix] -= a_edit[ix].mean()
print("Within-question score correlation:", round(float(np.corrcoef(a_orig, a_edit)[0,1]), 4))
print("Within-question score SD ratio:", round(float(a_edit.std()/a_orig.std()), 4))

head_diagnostics = {}
for layer, head in candidate_heads:
    stats, top_tokens = [], []
    for ex in range(8):
        tokens = [str(t) for t in baseline["attn_tokens_original"][ex]]
        seps = [i for i, t in enumerate(tokens) if t == "[SEP]"]
        qix = np.arange(1, seps[0])
        pix = np.arange(seps[0]+1, seps[-1])
        a = np.asarray(baseline["attn_original"][ex, layer, head], dtype=np.float32)
        stats.append([
            a[np.ix_(qix, pix)].sum(axis=1).mean(),
            a[np.ix_(pix, qix)].sum(axis=1).mean(),
            a[0, pix].sum(),
            a[0, seps].sum(),
            a[0, 0],
        ])
        top_tokens.append(tokens[int(np.argmax(a[0]))])
    ao = tensor(layer, "attention.output.dense.weight")
    vv = tensor(layer, "attention.self.value.weight")
    direction = torch.from_numpy(readout_direction.copy())
    path_projections = np.array([
        float((direction @ ao[:, 32*h:32*(h+1)] @ vv[32*h:32*(h+1)]).norm())
        for h in range(12)
    ])
    avg = np.mean(stats, axis=0)
    head_diagnostics[(layer, head)] = avg
    print((layer, head),
          "Q->P/P->Q/CLS->P/CLS->SEP/CLS->CLS", [round(float(x), 3) for x in avg],
          "readout-path relative to layer mean", round(float(path_projections[head]/path_projections.mean()), 3),
          "top CLS tokens", top_tokens)

with NOTES.open("a", encoding="utf-8") as f:
    f.write("\nCandidate A saves 23.84% FLOPs, with one selection regression and two gains. "
            "Whole layers retained because CLS similarity alone does not establish token-level redundancy.\n")
    f.write("Attention candidates examined using both directions of cross-segment attention, CLS targets, "
            "weight-path norms, and static readout projection. Small attention sample is hypothesis-generating only.\n")
