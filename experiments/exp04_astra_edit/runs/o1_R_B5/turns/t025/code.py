# Hypothesis: (pass 4) zeroing the 87 L4 coverage neurons' output columns lifts bridge golds vs beating distractors; scores-only pass, saved immediately.
S_cov = np.load(WORK / "S_cov_L4.npy"); W_now = load("bert.encoder.layer.4.output.dense.weight"); W_o = load_original("bert.encoder.layer.4.output.dense.weight")
chg_cols = np.where((W_now != W_o).any(0).numpy())[0]
print("k=%d | changed cols == S_cov: %s | zeroed: %s" % (len(S_cov), set(chg_cols) == set(S_cov.tolist()), bool((W_now[:, torch.as_tensor(S_cov)] == 0).all())))
NOTES.write_text("""# Notes: cross-encoder weight edit for multi-hop gold ranking
Split: A = even question index (choose), B = odd (confirm). Metrics on practice: tok@85/90/95 (tok length proxy), both-gold@3, pooled AUC hard-gold vs distractors.
Baseline: A tok90 945.5 b3 .58 AUC .720 | B tok90 963.1 b3 .51 AUC .641.
## Diagnosis (pass 1, original, all 1994 paragraphs, cls+mlp+attn)
- Hard (min-score) gold in bridge questions = answer-bearing 2nd-hop paragraph: has_ans .80, overlap .42, title_in_q .34, rank 4.5.
- Score within question tracks lexical match: corr overlap .60, title .46. Final [CLS] ~1-D (gold dir cos .96 with readout); no unused gold direction at L6 -> pooler edits pointless.
- Last layer: 7/12 heads at [CLS] read question tokens (coverage); h0/h6 read paragraph body.
- Layer trace along per-layer relevance axis (B): hard-minus-beating gap +.40sd (C2) -> +.08 (C3) -> -.22 (C4) -> -.61 (C5) -> -.66 (C6); layers 2-4 build coverage that favours high-overlap distractors.
- L4 MLP direct attribution to hard-minus-beating: -0.61 (A) / -0.70 (B); per-neuron attributions replicate (corr .98). Anti-hard neurons = coverage (corr body_ov .45).
- Length: model already penalises length (corr -.27); not used (dataset artifact).
- Title neurons: replicate but ~0 leverage on relevance axis.
## Passes
- P2: L5 heads 0,6 x2 -> null (B tok90 -8.8 +- 14.5). Dropped.
- P3: lost to kernel timeout.
- P4: L4 coverage neurons (87, chosen on A: 60% of negative hard-beating attribution) output columns x0.
""")
out4 = forward(items=items, model="edited")
S4 = np.asarray(out4["scores_edited"], float); np.save(WORK / "scores_cov_a1.npy", S4); print("saved; passes_left:", out4["passes_left"])
for sp, qs in [("A", qA), ("B", qB)]: print(sp, "orig", ev(S0, qs), "\n  cov a=1", ev(S4, qs), "| tok90 Δ", boot(S4, qs), "| tok85 Δ", boot(S4, qs, rc=0.85))
