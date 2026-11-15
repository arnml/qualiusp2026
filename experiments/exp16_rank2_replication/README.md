# exp16: rank-2 replication on 2WikiMultiHopQA and HotpotQA

**Question.** Is the pure conditioning change (Δ = iter − ctrl, matched training pairs) a rank-2 change on datasets other than MuSiQue?
Protocol: [`PROTOCOL.md`](PROTOCOL.md) (pre-registered at `353fcd8`, Codex review in `../reviews/codex_exp16_*`, deviation 1: A1 may be loaded from either Google account's Drive, sha check unchanged).

**Run.** Colab free T4 on the second Google account (`authuser=1`), notebook [`exp16_colab.ipynb`](exp16_colab.ipynb) at `2f52006`, code pinned to `353fcd8`. torch 2.14.1+cu130, transformers 5.22.1. Six models trained from A1 (sha `605c986e…`): iter, ctrl and ctrl seed 1 per dataset, 1,000 training questions each (2Wiki 19,028 pairs / 1,190 steps; HotpotQA 14,966 pairs / 936 steps; 2.4–4.8 min per model). Training-file sha256 values match Hugging Face's LFS checksums. HotpotQA splits exclude the 1,400 IDs used in exp01–07.

Raw outputs (models, per-question rankings) stay on that account's Drive (`MyDrive/qualiusp2026/exp16_353fcd8`). The repo keeps the analysis ([`results/results.json`](results/results.json)) and the per-question primary hits for every split and configuration ([`results/hits_compact.json`](results/hits_compact.json), `n:hex` bit strings in `split_ids.json` order); decoding them reproduces the test recoveries below.

## Results (primary metric: full-support recall@(|gold|+1))

| | 2WikiMultiHopQA | HotpotQA |
|---|---|---|
| Tuning gap (ctrl iterative → iter), gate ≥ 0.05 | 0.540 ✅ | 0.107 ✅ |
| Tuning recovery r1 / r2 / r4 / r8 | 0.877 / 0.901 / 0.963 / 0.969 | 0.844 / 0.844 / 0.875 / 1.000 |
| Random rank-2 (norm-matched) / seed-noise rank-2 (tuning) | 0.025 / −0.031 ✅ | 0.219 / −0.031 ✅ |
| Test (n = 1000): ctrl iterative / iter / rank-2 graft | 0.385 / 0.932 / 0.916 | 0.839 / 0.940 / 0.932 |
| **Test rank-2 recovery [95% CI]** | **0.971 [0.941, 1.000]** | **0.921 [0.816, 1.028]** |
| Test gap CI | [0.516, 0.578] | [0.079, 0.123] |
| Recovery at recall@5 (secondary) | 0.972 | 0.979 |
| Iterative queries truncated at 512 tokens | 28.2% | 6.1% |
| **Verdict** | **holds** | **holds** |

**Replication: yes** (both datasets hold, the pre-registered condition). With MuSiQue (exp12: 0.868 [0.777, 0.958]) the rank-2 result now holds on three multi-hop datasets.

## Descriptive (not tested)

- Iterative vs one-shot (iter vs ctrl one-shot): 2Wiki −0.011 (McNemar p = 0.32), HotpotQA −0.004 (p = 0.72). As on MuSiQue overall, iteration does not beat one-shot on these mostly two-hop datasets. 2Wiki by type: inference +0.19, bridge-comparison −0.13.
- Other ranks on test: 2Wiki r1 0.91, r4 0.99, r8 0.99; HotpotQA r1 0.84, r4 0.98, r8 0.99. A plateau from rank 2, rank 1 already most of it.
- The HotpotQA random control (0.219) is higher than on the other datasets because the gap is small (0.107): a few questions move the ratio. It passes the pre-registered rule (< half of the rank-2 recovery).

## Caveats

- One draw of the random and noise controls, on tuning data.
- On 2Wiki the ctrl model collapses when run iteratively (0.385), so the gap is large and the recovery ratio is easy to estimate; 28% of iterative queries are truncated.
- The skill is the same (conditioning on selected paragraphs); the replication is across datasets, not across model families.
