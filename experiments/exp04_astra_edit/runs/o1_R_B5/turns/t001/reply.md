I'll start by inspecting the environment and practice data before spending any forward passes.


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: none yet — inventory the data, tensors and notes before spending any forward pass.
import numpy as np, collections, json
print("WORK:", WORK, "| NOTES exists:", NOTES.exists())
if NOTES.exists(): print(NOTES.read_text()[:2000])
print("n tensors:", len(NAMES)); print([n for n in NAMES if "layer.5" in n or "pooler" in n or "classifier" in n])
print("n questions:", len(practice), "types:", collections.Counter(q["type"] for q in practice))
npar = [len(q["paragraphs"]) for q in practice]; print("paragraph counts:", collections.Counter(npar), "total:", sum(npar))
q = practice[0]; print(json.dumps({k: q[k] for k in ["question","answer","type","gold"]}, indent=0))
for p, g in zip(q["paragraphs"], q["gold"]): print(g, p[:150].replace("\n"," "))
lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for q in practice for p in q["paragraphs"]]
print("para token len: mean %.1f, median %d, max %d" % (np.mean(lens), np.median(lens), max(lens)))
print("edited():", edited())
</parameter>
</invoke>

<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: (pre-pass) define overlap/bridge features + metrics, then spend pass 1 on a full diagnostic of the original model.
import re, torch
STOP = set("the a an of in on at to for by with from and or is was are were be been which what who whom whose when where how did does do that this as it its his her their they he she than also other""".split())
def words(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower())]
def content(s): return set(w for w in words(s) if w not in STOP and len(w) > 1)
items, feats = [], []
for qi, q in enumerate(practice):
    qw = content(q["question"]); ans = q["answer"].lower().strip()
    for pi, p in enumerate(q["paragraphs"]):
        title, _, body = p.partition(". ")
        pw = content(p); tw = content(title)
        items.append((qi, pi))
        feats.append(dict(qi=qi, pi=pi, gold=q["gold"][pi], type=q["type"],
            overlap=len(qw & pw)/max(1,len(qw)),                     # frac of question content words in paragraph
            title_in_q=len(tw & qw)/max(1,len(tw)),                    # frac of title words in question
            has_ans=int(ans not in ("yes","no") and ans in p.lower()),
            ntok=len(tok(p, add_special_tokens=False)["input_ids"])))
import pandas as pd
F = pd.DataFrame(feats); print(F.shape, F.gold.sum())
print(F.groupby(["type","gold"])[["overlap","title_in_q","has_ans","ntok"]].mean().round(3))
# Split: even question index = A (choose edits), odd = B (confirm)
F["split"] = np.where(F.qi % 2 == 0, "A", "B")
out1 = forward(items=items, model="original", cls=True, mlp=True, attn_n=8)
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in out1.items() if k != "attn_tokens"})
np.save(WORK/"scores_orig.npy", out1["scores_original"]); F.to_pickle(WORK/"feats.pkl")
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: the original model fails mainly on low-overlap "bridge" gold paragraphs; define the metrics and verify the baseline.
S0 = out1["scores_original"].astype(float); F["s0"] = S0
def metrics(F, s, recall=0.90):
    F = F.assign(s=s); res = {}
    g = F.groupby("qi")
    # both-gold@3
    b3 = g.apply(lambda d: d.sort_values("s", ascending=False).head(3).gold.sum() == d.gold.sum())
    mingold = g.apply(lambda d: d.loc[d.gold==1, "s"].min())
    # threshold so that `recall` fraction of questions keep both golds (on this set)
    t = np.quantile(mingold.values, 1-recall, method="lower")
    def kept(d):
        k = d[d.s >= t]
        return (k if len(k) else d.nlargest(1, "s")).ntok.sum()
    tokens = g.apply(kept)
    return dict(b3=b3.mean(), tokens=tokens.mean(), total=g.ntok.sum().mean()), b3, mingold
for sp in ["A","B","all"]:
    sub = F if sp=="all" else F[F.split==sp]
    m,_,_ = metrics(sub, sub.s0); print(sp, {k: round(v,3) for k,v in m.items()})
# per-question: which gold is the min-gold (the one setting the threshold)?
F["rank0"] = F.groupby("qi").s0.rank(ascending=False)
G = F[F.gold==1].copy()
G["is_min"] = G.groupby("qi").s0.transform("min") == G.s0
print(G.groupby(["type","is_min"])[["overlap","title_in_q","has_ans","rank0","s0"]].mean().round(2))
D = F[F.gold==0]
print("distractor s0 mean %.2f | gold s0 mean %.2f" % (D.s0.mean(), G.s0.mean()))
print("within-question corr(s0, overlap): %.3f" % F.assign(so=F.s0-F.groupby('qi').s0.transform('mean'), ov=F.overlap-F.groupby('qi').overlap.transform('mean'))[['so','ov']].corr().iloc[0,1])
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: a residual direction at [CLS] tracks lexical overlap; find (on split A only) where gold-vs-distractor and overlap directions live, layer by layer.
C = out1["cls_original"]; M = out1["mlp_original"]          # C: (N, 7, 384)  M: (N, 6, 1536)
A = (F.split=="A").values; gold = F.gold.values.astype(bool)
qi = F.qi.values
def center_q(X):  # subtract per-question mean (within-question analysis)
    Xc = X.copy()
    for q in np.unique(qi): Xc[qi==q] -= X[qi==q].mean(0)
    return Xc
ovc = center_q(F.overlap.values[:,None])[:,0]; s0c = center_q(S0[:,None])[:,0]
summ = []
for L in range(7):
    X = center_q(C[:, L]); XA = X[A]
    d_gold = XA[gold[A]].mean(0) - XA[~gold[A]].mean(0)               # diff of means gold vs distractor (within q)
    d_ov = (XA * ovc[A,None]).mean(0) / ovc[A].var()                    # covariance with overlap (regression slope per dim, not fit into weights)
    cos = d_gold @ d_ov / (np.linalg.norm(d_gold)*np.linalg.norm(d_ov))
    # projection on held-out B: does d_gold separate gold in B?
    pB = X[~A] @ d_gold; gB = gold[~A]
    summ.append((L, round(np.linalg.norm(d_gold),3), round(cos,3), round((pB[gB].mean()-pB[~gB].mean())/pB.std(),3)))
print("layer, |d_gold|, cos(d_gold,d_overlap), B-separation(d_gold) in sd:"); [print(s) for s in summ]
# How does the final score relate? corr of within-q score with projections at final layer
X6 = center_q(C[:,6]); 
print("corr(s0c, ovc) all=%.3f" % np.corrcoef(s0c, ovc)[0,1])
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: final-layer [CLS] contains a gold signal the readout under-uses; check whether that gold direction is (near-)orthogonal to the readout and what score/overlap/answer info it carries.
Wp = load_original("bert.pooler.dense.weight").numpy(); wc = load_original("classifier.weight").numpy()[0]
r = Wp.T @ wc; r /= np.linalg.norm(r)                       # linear readout direction (weights only, tanh ignored)
XA6 = X6[A]
d6 = XA6[gold[A]].mean(0) - XA6[~gold[A]].mean(0); d6n = d6/np.linalg.norm(d6)
print("cos(readout r, d_gold_L6) = %.3f" % (r @ d6n))
# projections: compare score-like readout projection vs gold projection on split B
for name, v in [("readout r", r), ("d_gold", d6n)]:
    p = X6 @ v
    print(f"{name:10s} corr w/ s0c={np.corrcoef(p, s0c)[0,1]:.3f}  w/ overlap={np.corrcoef(p, ovc)[0,1]:.3f}  "
          f"w/ has_ans={np.corrcoef(p, center_q(F.has_ans.values[:,None].astype(float))[:,0])[0,1]:.3f}  "
          f"B gold sep(sd)={((p[~A][gold[~A]].mean()-p[~A][~gold[~A]].mean())/p[~A].std()):.3f}")
# component of d_gold orthogonal to r
d_perp = d6n - (d6n@r)*r; d_perp /= np.linalg.norm(d_perp)
p = X6 @ d_perp
print("d_perp: B gold sep(sd)=%.3f, corr w/ s0c=%.3f" % ((p[~A][gold[~A]].mean()-p[~A][~gold[~A]].mean())/p[~A].std(), np.corrcoef(p, s0c)[0,1]))
# Within-question: on split B, among min-gold (hard) paragraphs, is d_perp higher than distractors?
hard = (F.gold==1) & (F.s0 == F.groupby('qi').s0.transform(lambda s: s[F.loc[s.index,'gold']==1].min()))
print("B hard-gold vs distractor sep along d_perp: %.3f sd" % ((p[(~A)&hard.values].mean()-p[(~A)&~gold].mean())/p[~A].std()))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: specific layer-5 (last) MLP neurons at [CLS] carry the gold-vs-distractor signal independent of the score; find them on split A and check they replicate on B.
M5 = center_q(M[:, 5]); M4 = center_q(M[:, 4])
Wo5 = load_original("bert.encoder.layer.5.output.dense.weight").numpy()     # (384, 1536)
g5 = load_original("bert.encoder.layer.5.output.LayerNorm.weight").numpy()
def neuron_table(Mc, Wo, gamma):
    sd = Mc.std(0) + 1e-6
    dA = (Mc[A][gold[A]].mean(0) - Mc[A][~gold[A]].mean(0)) / sd          # standardized gold-vs-distractor diff (split A)
    dB = (Mc[~A][gold[~A]].mean(0) - Mc[~A][~gold[~A]].mean(0)) / sd       # same on split B (for replication only)
    cs = np.array([np.corrcoef(Mc[:, n], s0c)[0, 1] for n in range(Mc.shape[1])])   # corr with current score
    co = np.array([np.corrcoef(Mc[:, n], ovc)[0, 1] for n in range(Mc.shape[1])])   # corr with overlap
    wr = (Wo * gamma[:, None]).T @ r                                       # weight-only: neuron out-vector on readout
    return pd.DataFrame(dict(dA=dA, dB=dB, corr_s=cs, corr_ov=co, w_readout=wr, sd=sd))
T5 = neuron_table(M5, Wo5, g5)
print("corr(dA, dB) across neurons: %.3f" % np.corrcoef(T5.dA, T5.dB)[0,1])
print("top gold-promoting neurons (by A):"); print(T5.sort_values("dA", ascending=False).head(10).round(3))
print("top distractor-promoting neurons (by A):"); print(T5.sort_values("dA").head(10).round(3))
# Neurons whose readout contribution is positive but which point AWAY from gold: candidate 'overlap' writers
T5["contrib_sign"] = np.sign(T5.w_readout)
print("neurons with corr_ov>0.3:", (T5.corr_ov>0.3).sum(), "| with dA<-0.3:", (T5.dA<-0.3).sum())
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: the gold signal lives in a distributed residual direction (not a few neurons); look at layer-wise attention & last-layer heads via CLS deltas, and quantify the candidate pooler-row edit budget.
# 1) How much does the score depend on the readout vs. d_perp? Characterize distractors that beat the hard gold (split A only).
FA = F[F.split=="A"].copy()
hardA = FA[(FA.gold==1)].groupby("qi").s0.min()
FA["hard_gold_s"] = FA.qi.map(hardA)
beat = FA[(FA.gold==0) & (FA.s0 > FA.hard_gold_s)]
print("A: distractors outscoring hard gold: %d of %d distractors" % (len(beat), (FA.gold==0).sum()))
print("their mean overlap %.2f vs other distractors %.2f; hard golds %.2f" % (beat.overlap.mean(),
      FA[(FA.gold==0)&~FA.index.isin(beat.index)].overlap.mean(),
      FA[(FA.gold==1)&(FA.s0==FA.hard_gold_s)].overlap.mean()))
# 2) Pooler geometry: which pooler units matter for classifier and are not saturated? (weights + recorded tanh-free stats only)
Wp = load_original("bert.pooler.dense.weight").numpy(); bp = load_original("bert.pooler.dense.bias").numpy()
print("|wc| sorted top 10:", np.round(np.sort(np.abs(wc))[::-1][:10],3), " median |wc| %.3f" % np.median(np.abs(wc)))
# projection of d_perp through pooler rows: how strongly does each pooler unit already read d_perp?
rd = Wp @ d_perp
print("corr over pooler units between wc and Wp@d_perp: %.3f" % np.corrcoef(wc, rd)[0,1])
print("budget: rows editable = %d of 384 (each 384 weights)" % (108000//384))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: d_perp is a "multi-hop link / answer-bearing" feature present at [CLS] that the pooler barely reads; build a targeted pooler-row edit that adds lambda*d_perp to the units the classifier reads most, then dose-response.
import numpy as np, torch
# choose K pooler rows with the largest |classifier weight| (weight-only choice; independent of practice labels)
K = 270                                   # 270*384 = 103,680 weights  < 108,000
rows = np.argsort(-np.abs(wc))[:K]
# per-unit gain: put the extra drive in the direction that raises the score: sign(wc[j]) * |wc[j]| / ||wc[rows]||
c = wc[rows] / np.linalg.norm(wc[rows])
# scale reference: typical spread of the readout pre-activation along r vs. along d_perp (within-question SDs, recorded stats)
sd_r  = (X6 @ r).std(); sd_dp = (X6 @ d_perp).std()
print("within-q SD along r: %.3f | along d_perp: %.3f | ||Wp|| row mean %.3f" % (sd_r, sd_dp, np.linalg.norm(Wp,axis=1).mean()))
def make_edit(lam):
    W = load_original("bert.pooler.dense.weight").clone()
    delta = lam * np.outer(c, d_perp)                         # rank-1, only K rows touched
    W[rows] += torch.tensor(delta, dtype=W.dtype)
    return W
# store direction + metadata so the edit is reproducible
np.savez(WORK/"edit_dirs.npz", d_perp=d_perp, r=r, rows=rows, c=c)
for lam in [0.5, 1.0, 2.0]:
    W = make_edit(lam); print("lam", lam, "changed entries:", int((W != load_original("bert.pooler.dense.weight")).sum()),
                              "rel Frobenius change %.3f" % ((W - load_original("bert.pooler.dense.weight")).norm()/load_original("bert.pooler.dense.weight").norm()))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: adding lam*d_perp to the top-|wc| pooler rows raises gold (esp. bridge) paragraphs relative to distractors; test lam=2.0 on all items (model="both"), judge on split B.
save_edit("bert.pooler.dense.weight", make_edit(2.0)); print(edited())
out2 = forward(items=items, model="both", cls=False)
S2 = out2["scores_edited"].astype(float); print("orig scores identical to pass1:", np.allclose(out2["scores_original"], S0, atol=1e-4), "passes_left:", out2["passes_left"])
np.save(WORK/"scores_lam2.npy", S2)
for sp in ["A","B"]:
    sub = F[F.split==sp]
    m0,b0,_ = metrics(sub, sub.s0); m2,b2,_ = metrics(sub, S2[sub.index])
    d = (b2.astype(int)-b0.astype(int))
    print(sp, "orig", {k: round(v,3) for k,v in m0.items()}, "| lam2", {k: round(v,3) for k,v in m2.items()},
          "| b3 per-q: +%d -%d (se of diff %.3f)" % ((d>0).sum(), (d<0).sum(), d.std()/np.sqrt(len(d))))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: the lam=2.0 gain is mainly from lifting the hard (min) gold paragraphs; inspect per-type effects on split B and bootstrap the token gain for an SE.
rng = np.random.default_rng(0)
def boot_tokens(sub, s_a, s_b, n=1000):
    qs = sub.qi.unique(); diffs = []
    for _ in range(n):
        pick = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([sub.index[sub.qi==q].values for q in pick])
        # relabel duplicates so groupby treats them separately
        tmp = sub.loc[idx].copy(); tmp["qi"] = np.repeat(np.arange(len(pick)), [ (sub.qi==q).sum() for q in pick])
        tmp = tmp.reset_index(drop=True)
        diffs.append(metrics(tmp, s_b[idx])[0]["tokens"] - metrics(tmp, s_a[idx])[0]["tokens"])
    return np.mean(diffs), np.std(diffs)
subB = F[F.split=="B"]
mu, se = boot_tokens(subB, S0, S2, n=300)
print("B tokens@90 change (lam2 - orig): %.1f ± %.1f (bootstrap SE)" % (mu, se))
for t in ["bridge","comparison"]:
    sub = subB[subB.type==t]
    m0,_,_ = metrics(sub, sub.s0); m2,_,_ = metrics(sub, S2[sub.index])
    print(t, "n=%d" % sub.qi.nunique(), "b3 %.3f -> %.3f" % (m0["b3"], m2["b3"]))
# hard-gold rank improvement on B
FB = subB.assign(s2=S2[subB.index]); FB["r2"] = FB.groupby("qi").s2.rank(ascending=False)
hg = FB[FB.gold==1].groupby("qi")[["rank0","r2"]].max()
print("B worst-gold rank: mean %.2f -> %.2f; improved %d, worse %d" % (hg.rank0.mean(), hg.r2.mean(), (hg.r2<hg.rank0).sum(), (hg.r2>hg.rank0).sum()))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: if d_perp is the causal lever, the effect should grow with strength; run lam=4.0 (pass 3) to get a second dose point.
save_edit("bert.pooler.dense.weight", make_edit(4.0)); print(edited())
out3 = forward(items=items, model="edited")
S4 = out3["scores_edited"].astype(float); np.save(WORK/"scores_lam4.npy", S4); print("passes_left:", out3["passes_left"])
for sp in ["A","B"]:
    sub = F[F.split==sp]
    for nm, s in [("orig", sub.s0.values), ("lam2", S2[sub.index]), ("lam4", S4[sub.index])]:
        m,_,_ = metrics(sub, s); print(sp, nm, {k: round(v,3) for k,v in m.items()})
mu4, se4 = boot_tokens(subB, S0, S4, n=300); print("B tokens@90 change (lam4 - orig): %.1f ± %.1f" % (mu4, se4))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: the response to lam saturates (tanh) and turns over past ~4; check where tanh saturation bites using split-A per-question changes, and record findings in NOTES before deciding the final strength.
subA = F[F.split=="A"]
for nm, s in [("lam2", S2), ("lam4", S4)]:
    mu, se = boot_tokens(subA, S0, s, n=200)
    print("A tokens@90 change (%s - orig): %.1f ± %.1f" % (nm, mu, se))
# per-type on B for lam4
for t in ["bridge","comparison"]:
    sub = subB[subB.type==t]
    print(t, "b3 orig %.3f lam2 %.3f lam4 %.3f" % (metrics(sub, sub.s0)[0]["b3"], metrics(sub, S2[sub.index])[0]["b3"], metrics(sub, S4[sub.index])[0]["b3"]))
NOTES.write_text(f"""# Notes — cross-encoder weight edit (multi-hop gold paragraph ranking)

## Split
- A = even question index (used to choose direction & strength), B = odd question index (held out for confirmation).

## Diagnosis (pass 1, original model, all {len(items)} practice paragraphs, cls+mlp)
- Hard (min-score) gold paragraphs are mostly bridge paragraphs with low lexical overlap with the question; distractors that outrank
  them have high overlap. Within-question corr(score, overlap) is strongly positive.
- Final-layer [CLS] has a gold-vs-distractor direction d_gold (diff of within-question means, split A).
  Its component orthogonal to the linear readout r = Wp^T wc ("d_perp") separates gold from distractors on split B,
  including hard golds, but the pooler barely reads it (cos(r, d_gold) small).

## Edit
- bert.pooler.dense.weight: rows = top-270 |classifier weight| units; W[rows] += lam * outer(c, d_perp),
  c = wc[rows]/||wc[rows]||. 270*384 = 103,680 changed weights (< 108,000).
- Pass 2: lam=2 (both). Pass 3: lam=4 (edited).
""")
print("notes written")
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: lam=3 sits between the two dose points; confirm on B with a single pass (pass 4) and keep pass 5 in reserve.
save_edit("bert.pooler.dense.weight", make_edit(3.0)); print(edited())
out4 = forward(items=items, model="edited")
S3 = out4["scores_edited"].astype(float); np.save(WORK/"scores_lam3.npy", S3); print("passes_left:", out4["passes_left"])
for sp in ["A","B"]:
    sub = F[F.split==sp]
    for nm, s in [("orig", sub.s0.values), ("lam2", S2[sub.index]), ("lam3", S3[sub.index]), ("lam4", S4[sub.index])]:
        m,_,_ = metrics(sub, s); print(sp, nm, {k: round(v,3) for k,v in m.items()})
mu3, se3 = boot_tokens(subB, S0, S3, n=300); print("B tokens@90 change (lam3 - orig): %.1f ± %.1f" % (mu3, se3))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: lam=4 is the best dose on A; set it as the submitted edit, save a copy, and spend the last pass on custom bridge pairs to check the mechanism generalizes beyond practice items.
save_edit("bert.pooler.dense.weight", make_edit(4.0)); print(edited())
torch.save(load("bert.pooler.dense.weight"), WORK/"pooler_lam4.pt")
# hand-written probes: same question, a bridge (answer-bearing, low overlap) paragraph vs a high-overlap distractor
pairs = [
 ("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "Shirley Temple. Shirley Temple Black was an American actress, singer, dancer, businesswoman, and diplomat who served as Chief of Protocol of the United States."),
 ("What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "Kiss and Tell (play). Kiss and Tell is a comedy play by F. Hugh Herbert about Corliss Archer, first produced on Broadway in 1943."),
 ("Which city is the birthplace of the director of the film Inception?",
  "Christopher Nolan. Christopher Edward Nolan is a British-American film director born in London, England."),
 ("Which city is the birthplace of the director of the film Inception?",
  "Inception (soundtrack). Inception is the soundtrack album to the film Inception, composed by Hans Zimmer and released in 2010."),
]
out5 = forward(pairs=pairs, model="both")
print("passes_left:", out5["passes_left"])
for (q,p), so, se_ in zip(pairs, out5["scores_original"], out5["scores_edited"]):
    print("orig %6.2f  edited %6.2f  | %s" % (so, se_, p[:60]))
</parameter>
</invoke>


<invoke name="mcp__kernel__python">
<parameter name="code">
# Hypothesis: (finalize) confirm the submitted state is lam=4, within budget, and update NOTES with the confirmed numbers.
print(edited())
W = load("bert.pooler.dense.weight"); W0 = load_original("bert.pooler.dense.weight")
print("changed entries:", int((W != W0).sum()), "| matches saved copy:", torch.allclose(W, torch.load(WORK/"pooler_lam4.pt")))
subA = F[F.split=="A"]
rows_out = []
for sp, sub in [("A", subA), ("B", subB)]:
    for nm, s in [("orig", S0), ("lam2", S2), ("lam3", S3), ("lam4", S4)]:
        m,_,_ = metrics(sub, s[sub.index]); rows_out.append((sp, nm, round(m["b3"],3), round(m["tokens"],1)))
tbl = pd.DataFrame(rows_out, columns=["split","edit","both_gold@3","tokens@90"]); print(tbl.to_string(index=False))
with open(NOTES, "a") as fh:
    fh.write("\n## Results (practice; A chose lam, B held out)\n" + tbl.to_string(index=False) + "\n")
    fh.write(f"\n- B tokens@90 change lam4 vs orig: {mu4:.1f} ± {se4:.1f} (bootstrap SE over questions)\n")
    fh.write("- Final submitted edit: lam=4 (copy at WORK/pooler_lam4.pt). Pass 5 = custom bridge/distractor probe pairs.\n")
print("notes updated")
</parameter>
</invoke>