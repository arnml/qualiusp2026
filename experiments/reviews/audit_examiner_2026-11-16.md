# Examiner audit: research contribution and method (2026-11-16)

Point of view: a qualification committee member judging contribution and method.
Scope: `quali/chapters/*.tex` at `66a6310`, checked against `experiments/HYPOTHESES.md` and the exp11, exp12, exp13, exp14 and exp16 reports and code.
Nothing in `quali/` or `bibli.bib` was edited. LaTeX was not rendered.

Severity: **Major** = a committee member would likely raise it in the exam. **Moderate** = weakens the text. **Minor** = wording or presentation.

---

## A. Contribution and novelty

### 1. Major: the research contribution is never stated as one sentence, and it comes after the practical results
- `00_abstract.tex:16-18` (and the resumo, `:5-7`): the abstract opens with token savings (practical). Then it gives the long-chain gains and the cascade. The rank-2 finding, which is the title's claim, comes fourth. It is closed by a vague sentence: "which characterizes what this fine-tuning adds to a reranker" (`:18`).
- `01_introduction.tex:36-41`: contribution 1 is the practical evaluation. The main scientific claim is merged into item 3 together with the long-chain results.
- **Fix.** Put one explicit contribution sentence near the start of the abstract's second paragraph. Then reorder the contributions as scientific, then supporting, then practical, then methodological. Suggested abstract sentence:

```latex
The main finding is structural: in a 22.7M-parameter cross-encoder, the weight change that lets the model use already-selected evidence in its query is functionally low-rank. Grafting only its rank-2 truncated SVD (with the one-dimensional parameters in full) onto a model trained on the same pairs without conditioning restores 86.8\% of the change's effect on MuSiQue, and 97.1\% and 92.1\% in a pre-registered replication on 2WikiMultiHopQA and HotpotQA, against at most 22\% for norm-matched random changes.
```

  Suggested `\section{Contributions so far}` (replaces `01_introduction.tex:36-41`):

```latex
\begin{enumerate}
  \item \textbf{Main contribution (RQ4).} Evidence that a retrieval behaviour, using already-selected evidence in the query, is carried by a functionally low-rank weight change in a small cross-encoder. The change is isolated with a pair-matched control. Its rank-2 truncation restores 87--97\% of the behaviour on three multi-hop datasets, against norm-matched random and seed-noise controls (Sections~\ref{sec:res_matched}, \ref{sec:res_replication}).
  \item \textbf{Where the behaviour pays off (RQ3).} With training data matched, conditioning-trained iteration does not significantly beat one-shot ranking overall. It does improve full-support recall on four-hop chains and answer F1 on questions with three or more hops. A router trained on per-question benefit, but not one trained on hop count, turns this into a cascade that improves on one-shot ranking (Sections~\ref{sec:res_long}, \ref{sec:res_f1}).
  \item \textbf{Practical consequence (RQ1, RQ2).} A token accounting of a 22.7M-parameter selector at fixed recall targets and with an LLM reader. It shows that on MuSiQue the ranker, not the stopping rule, limits cost (Sections~\ref{sec:res_selector}--\ref{sec:res_bottleneck}).
  \item \textbf{Method.} A pre-registered, adversarially reviewed protocol for an AI-assisted project, with a claim ledger that records nulls (Section~\ref{sec:protocol}).
\end{enumerate}
```

### 2. Major: "two directions" and "only two directions" understate what the graft adds
- `00_abstract.tex:16` ("adding back only two directions"), `:5` (resumo), `01_introduction.tex:30`, `06_plan.tex:9`.
- The code (`exp11_iteration_skill/graft.py:47-60`, the same in `exp12_matched_control/evaluate.py:44`) adds every one-dimensional delta in full: biases, LayerNorm and the 1×384 classifier. That is about 30,000 of the about 180,000 adapter parameters.
- No 1-D-only graft (a "rank-0" graft) was run. The random control keeps the full 1-D deltas and only randomizes the 2-D part. Its low recovery (7%, 3%) therefore bounds the 1-D contribution indirectly. The text never says so.
- **Fix.**
  - In `04_method.tex:58`, add: "The random control keeps the one-dimensional deltas in full, so it also bounds what those deltas alone contribute."
  - In the abstract, write "two directions per weight matrix (plus the bias and normalization changes)".
  - Add a rank-0 graft to W4 (cheap: one evaluation per dataset).

### 3. Major: by the document's own definition, the functional rank is 1 on two of the three datasets
- `04_method.tex:58` defines the selection rule as "the smallest rank with at least 80% recovery" on the tuning set.
- Under that rule the functional rank is:
  - MuSiQue: tuning r1 = 0.728, r2 = 0.851, so r\* = 2;
  - 2WikiMultiHopQA: tuning r1 = 0.877, so r\* = 1;
  - HotpotQA: tuning r1 = 0.844, so r\* = 1 (exp16 README).
- exp16 fixed r = 2 by protocol, so the replication itself is valid. But "the change is functionally rank 2" (title of `05_results.tex:39`, `06_plan.tex:57`) is threshold-dependent. `05_results.tex:110` admits that it is "an upper bound".
- **Fix.**
  - Define functional rank formally (finding 7) and state the claim as r\*\_{0.8} ≤ 2 on all three datasets, and = 1 on two of them.
  - Rename `\section{The iteration change is functionally rank 2}` to "The iteration change is functionally low-rank (rank at most 2)".
  - Report the rank-1 recoveries in Table `tab:replication` (one extra column).
  - Leaving this unclaimed also gives away a stronger result: rank 1 already recovers 73–91%, which matches the rank-1 behaviour literature (Turner, Soligo).

### 4. Moderate: positioning against related work is mostly correct, with one overclaim and one gap
- **Correct.**
  - LoRA and intrinsic dimension: a low-rank update is "not surprising" (`03_related_work.tex:26`).
  - PHLoRA is the same extraction operation (`:22`).
  - Nijasure et al. 2025: their rank-1 LoRA rerankers are trained for relevance, while this change is extracted and isolates an added behaviour (`:26`).
  - Beam Retrieval: the iterative reranker is not new (`:18`).
  - Perlman et al.: mid-layer attention locus (`:22`). This contrasts with the flatter layer profile found here (`05_results.tex:42`), and the contrast is worth one explicit sentence in the related-work chapter.
- **Overclaim.** `03_related_work.tex:41`, positioning table: "The same in a small encoder". Soligo et al. show a *mediating activation direction* whose ablation transfers across fine-tunes. This work shows only *functional sufficiency of a weight-space truncation*: no activation direction, no ablation, no transfer yet. Replace with: "A retrieval behaviour in a small encoder is restored by a rank-2 weight graft (sufficiency); mediation and transfer untested (W4, W5)."
- **Gap: possibly missing related work.** The delta-compression literature compresses fine-tuning deltas by low-rank or SVD and reports long-tailed spectra, for example BitDelta (Liu et al., 2024) and Delta-CoMe (Ping et al., 2024). A committee member who knows model merging may ask about it. Check these on their official pages before citing them; they were not verified here.
- **Unclaimed results that are new and supported:**
  1. **The 2×2 dissociation** (exp12, `05_results.tex:56`). Without the conditioning change the one-shot model collapses when run iteratively (0.41). Conditioning costs a little one-shot quality (0.685 vs 0.7075). The change is therefore *necessary* for iterative use as well as rank-2 *sufficient*. Only the sufficiency side is claimed.
  2. **Hop count vs benefit routing** (exp13 vs exp14). Predicting hop count fails and predicting benefit succeeds. This means "long chain" is not the operative variable for when iteration helps, which nuances the hypothesis in `01_introduction.tex:25`. Say so in Section 5.6 and in the hypothesis discussion.
  3. **The seed-noise control.** Grafting the rank-r part of a seed-to-seed difference is a negative control for functional-rank claims that the cited rank literature does not use. Claim it under contribution 4.

### 5. Moderate: RQ4 and the hypothesis promise "reusable" and "trained and shipped", which are untested
- `01_introduction.tex:20` ("can it be isolated as a small, reusable component"), `:25`, `00_abstract.tex:18` ("separable").
- Reuse (transfer, W5) and direct training (W1) are future work. "Separable" is supported only in the sense of grafting onto the matched base.
- **Fix.** Split RQ4 into RQ4a (structure, answered now) and RQ4b (can it be trained directly and reused across datasets and rerankers: W1, W5).

### 6. Minor: the random-control range "7–9%" mixes two different deltas
- `00_abstract.tex:16` and `01_introduction.tex:39` put "86.8% / about 85%" next to "random 7–9%".
- 7% is the random rank-2 control for Δ = M1-iter − M1-ctrl (exp12). 9% is the random rank-4 control for M1-iter − M1 (exp11).
- **Fix.** "against 7% for a random change of the same norm" (MuSiQue, matched delta).

---

## B. Mathematical presentation

### 7. Major: the SVD is used but never defined, and "functional rank" and "recovery" are not formal definitions
- `02_background.tex:25-30` jumps from LoRA to "its best rank-r approximation in the Frobenius norm is the truncated SVD". The following are all missing:
  - the decomposition itself;
  - orthonormality of U and V;
  - the ordering of the singular values (it is stated only in passing);
  - a definition of the Frobenius norm;
  - the statement of Eckart–Young, including its error formula;
  - the condition for a unique minimizer;
  - the link between Δ_r and the LoRA form BA, which makes the comparison with PHLoRA and W1 precise.
- "Functional rank" is informal (`02_background.tex:32`).
- `04_method.tex:58` says R is "recall@5", but exp16 uses recall@(|gold|+1). The definition must allow a dataset-specific metric.
- W4's "energy" and "stable rank" (`06_plan.tex:28`) are undefined.
- **Fix:** replace `02_background.tex:25-30` with the subsection below. Keep the intuition paragraph (`:32`) after it, minus its last two sentences, which move into Definition 2. The label `eq:svd` is kept because `04_method.tex:53` references it. Two notes on the LaTeX:
  - `amsthm` is loaded (`main.tex:8`), but no `definition` or `theorem` environment is declared. Either add the preamble lines below, or turn the definitions into bold run-in paragraphs.
  - The citation keys `golub2013matrix`, `mirsky1960symmetric` and `rudelson2007sampling` are new (finding 8).

Preamble (`main.tex`, after `\DeclareMathOperator*{\argmax}`):

```latex
\theoremstyle{plain}\newtheorem{theorem}{Theorem}[chapter]
\theoremstyle{definition}\newtheorem{definition}{Definition}[chapter]
\DeclareMathOperator{\rank}{rank}
```

Background subsection (`02_background.tex`, replacing lines 25-30):

```latex
Fine-tuning changes a pretrained weight matrix $W_0 \in \mathbb{R}^{m \times n}$ into $W_1 = W_0 + \Delta$. Several lines of work show that useful updates are often well approximated in a low-dimensional subspace: \citeonline{aghajanyan2021intrinsic} measure a low intrinsic dimension of fine-tuning, and LoRA \cite{hu2022lora} trains only a low-rank update $\Delta = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$ and small $r$. This section defines the tools used to describe a full update $\Delta$.

\subsection{Singular value decomposition}
\label{sec:bg_svd}

Let $p=\min(m,n)$. Every real matrix $\Delta\in\mathbb{R}^{m\times n}$ has a \emph{singular value decomposition} (SVD) \cite[Sec.~2.4]{golub2013matrix}
\begin{equation}
\Delta \;=\; U\Sigma V^\top \;=\; \sum_{i=1}^{p}\sigma_i\,u_i v_i^\top ,
\label{eq:svdfull}
\end{equation}
where $U=[u_1,\dots,u_p]\in\mathbb{R}^{m\times p}$ and $V=[v_1,\dots,v_p]\in\mathbb{R}^{n\times p}$ have orthonormal columns ($U^\top U=V^\top V=I_p$) and $\Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_p)$ holds the \emph{singular values} $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_p\ge 0$. The vectors $u_i$ and $v_i$ are the left and right singular vectors, and $\rank(\Delta)$ is the number of nonzero singular values. The Frobenius and spectral norms are
\begin{equation}
\|\Delta\|_F=\Big(\sum_{j,k}\Delta_{jk}^2\Big)^{1/2}=\Big(\sum_{i=1}^{p}\sigma_i^2\Big)^{1/2},
\qquad \|\Delta\|_2=\sigma_1 .
\end{equation}

For $r\le p$, the \emph{truncated SVD} keeps the $r$ largest terms:
\begin{equation}
\Delta_r=\sum_{i=1}^{r}\sigma_i\,u_i v_i^\top=U_r\Sigma_r V_r^\top ,
\label{eq:svd}
\end{equation}
with $U_r$, $V_r$ the first $r$ columns of $U$, $V$ and $\Sigma_r=\operatorname{diag}(\sigma_1,\dots,\sigma_r)$. Writing $B=U_r\Sigma_r$ and $A=V_r^\top$ gives $\Delta_r=BA$, the LoRA form: a rank-$r$ truncation is a rank-$r$ adapter with $r(m+n)$ numbers instead of $mn$. Extracting adapters from fine-tuned checkpoints in this way is the operation of PHLoRA \cite{vasani2025phlora}.

\begin{theorem}[Eckart--Young]
For every $X\in\mathbb{R}^{m\times n}$ with $\rank(X)\le r$,
\[
\|\Delta-\Delta_r\|_F\;\le\;\|\Delta-X\|_F,
\qquad
\|\Delta-\Delta_r\|_F^2=\sum_{i=r+1}^{p}\sigma_i^2 .
\]
\end{theorem}
The result is due to \citeonline{eckart1936approximation}; \citeonline{mirsky1960symmetric} extended it to the spectral norm and to every unitarily invariant norm. The minimizer is unique when $\sigma_r>\sigma_{r+1}$.

Two \emph{spectral} summaries describe how concentrated $\Delta$ is. The \emph{energy} retained at rank $r$ is $E_r(\Delta)=\sum_{i\le r}\sigma_i^2/\sum_{i\le p}\sigma_i^2 = 1-\|\Delta-\Delta_r\|_F^2/\|\Delta\|_F^2$, and the \emph{stable rank} is $\|\Delta\|_F^2/\|\Delta\|_2^2\le\rank(\Delta)$ \cite{rudelson2007sampling}. Both depend only on the matrix. This work's claims are instead \emph{functional}: they ask how much of a model's behaviour a truncation preserves (Definition~\ref{def:functional_rank}). A change can be functionally rank 2 while most of its energy lies in other directions. Spectral measures of fine-tuning updates also need a noise baseline before they are interpreted \cite{altaf2026noisebaseline}.
```

Method definitions (`04_method.tex`, replacing the sentence after Equation `eq:recovery`, line 58):

```latex
\begin{definition}[Rank-$r$ graft and recovery]
Let $\Delta^{(\ell)}=W^{(t,\ell)}-W^{(b,\ell)}$ for every parameter tensor $\ell$. The rank-$r$ graft $g_r$ sets each two-dimensional tensor to $W^{(b,\ell)}+\Delta^{(\ell)}_r$ (Equation~\ref{eq:svd}) and each one-dimensional tensor to $W^{(b,\ell)}+\Delta^{(\ell)}$. Its recovery $\mathrm{rec}(g_r)$ is Equation~\ref{eq:recovery}, where $R$ is the dataset's primary full-support recall (recall@5 on MuSiQue, recall@$(|\text{gold}|+1)$ on HotpotQA and 2WikiMultiHopQA) under the target's inference mode.
\end{definition}

\begin{definition}[Functional rank]
\label{def:functional_rank}
For a threshold $\theta\in(0,1]$, the functional rank of $\Delta$ is
$r^*_\theta(\Delta)=\min\{\,r\in\{1,2,4,8,16\}:\mathrm{rec}(g_r)\ge\theta\,\}$, measured on the tuning set. This work uses $\theta=0.8$. A claim ``$\Delta$ is functionally rank $r$'' means $r^*_{0.8}(\Delta)\le r$, confirmed by a held-out recovery with a lower confidence bound above a pre-registered value.
\end{definition}

As controls, the two-dimensional part $\Delta_r$ is replaced by (i) a random rank-$r$ matrix $GH^\top/\|GH^\top\|_F\cdot\|\Delta_r\|_F$ with Gaussian $G\in\mathbb{R}^{m\times r}$ and $H\in\mathbb{R}^{n\times r}$, or (ii) the rank-$r$ truncation of the difference between two base models trained with different seeds (seed noise). In both controls the one-dimensional deltas are kept in full, so the controls also bound what those deltas alone recover.
```

### 8. New bibliography entries (metadata checked; add to the root `bibli.bib` and copy to `quali/bibli.bib`)

```bibtex
@book{golub2013matrix,
  title     = {Matrix Computations},
  author    = {Golub, Gene H. and Van Loan, Charles F.},
  edition   = {4},
  series    = {Johns Hopkins Studies in the Mathematical Sciences},
  publisher = {Johns Hopkins University Press},
  address   = {Baltimore},
  year      = {2013},
  isbn      = {978-1-4214-0794-4}
}

@article{mirsky1960symmetric,
  title     = {Symmetric Gauge Functions and Unitarily Invariant Norms},
  author    = {Mirsky, L.},
  journal   = {The Quarterly Journal of Mathematics},
  volume    = {11},
  number    = {1},
  pages     = {50--59},
  year      = {1960},
  doi       = {10.1093/qmath/11.1.50}
}

@article{rudelson2007sampling,
  title     = {Sampling from Large Matrices: An Approach through Geometric Functional Analysis},
  author    = {Rudelson, Mark and Vershynin, Roman},
  journal   = {Journal of the ACM},
  volume    = {54},
  number    = {4},
  pages     = {21},
  year      = {2007},
  doi       = {10.1145/1255443.1255449}
}
```

How each entry was verified:
- **Mirsky.** Checked on the OUP article page (academic.oup.com, DOI 10.1093/qmath/11.1.50).
- **Rudelson and Vershynin.** Checked through Crossref for the ACM DOI. J. ACM uses article numbers; 21 is the article number.
- **Golub and Van Loan.** The JHU Press page (press.jhu.edu/books/title/10678/matrix-computations) returned HTTP 403 to the fetcher. The ISBN 978-1-4214-0794-4 (hardcover, 2013, 4th ed.) is the one shown for that page in search results and on the book's copyright page; the e-book ISBN is 978-1-4214-0859-0.
- **Before adding Golub and Van Loan,** open the publisher page in a browser, as AGENTS.md requires. In the 4th edition, Sec. 2.4 is "The Singular Value Decomposition", and the Eckart–Young statement is Theorem 2.4.8. Confirm the section number in the copy you use.

---

## C. Research protocol and threats to validity

### 9. Major: the planned locked-set confirmation has no data source, and MuSiQue dev is exhausted
- `06_plan.tex:38` and `:90` and HYPOTHESES.md say: "re-tested on a locked set". But exp13 used "all unused dev questions" (`dev_fresh`, 1,017), so no MuSiQue-Ans dev question is left.
- MuSiQue-Ans train had 789 unused four-hop questions at exp13, and exp13 used 400 of them.
- exp14 then drew about 7,000 of the 13,538 remaining train questions, of which about 3–4% are four-hop. So roughly 200 four-hop questions remain unused.
- At n ≈ 200, the MDE for the four-hop claim (+0.055) is about 0.10. A locked confirmation of the four-hop claim on MuSiQue would be underpowered.
- **Fix.** In Chapter 6, name the locked set now:
  - freeze and commit the IDs, with a sha, of all still-unused MuSiQue-Ans train questions;
  - state the power for each headline claim;
  - for the four-hop claim, either pool with 2WikiMultiHopQA four-support questions (declared as a different population), or downgrade the locked test to the claims it can power: the rank-2 recovery, which has large effects, and the benefit cascade.
  - Add the exhausted dev split as a threat.

### 10. Major: multiplicity is acknowledged but its consequence is not stated
- `06_plan.tex:38`: "across experiments no correction is applied". HYPOTHESES.md counts about 12 confirmatory primaries.
- With a Bonferroni threshold of 0.05/12 ≈ 0.0042:
  - the four-hop recall claim (p = 0.035, `05_results.tex:61`) **does not survive**;
  - the rank-2 recoveries (CIs far from the thresholds), exp15 (CI [0.041, 0.112]) and exp14 (p = 5×10⁻⁶) do survive.
- A committee member will compute this.
- **Fix.** Add one sentence to the threats, and a "survives Holm across the family" column, or a footnote, to Table `tab:claims`:

```latex
\item \textbf{Multiplicity.} About twelve experiments each had one pre-registered primary test; no correction is applied across experiments. Under a Holm correction over this family, the rank-2 recoveries, the answer-F1 gain and the benefit cascade remain significant, but the four-hop recall gain ($p=0.035$) does not. It is therefore treated as supported but awaiting confirmation on the locked set.
```

### 11. Moderate: deviations are described in the results, not in the protocol section
- The exp14 re-training is explained well at `05_results.tex:65`. But Section `sec:protocol` (`04_method.tex:63`) says only that "deviations were recorded".
- **Fix.** Add a short paragraph, or a "Deviations" column in the appendix index (finding 14), listing each deviation that touches the evidence:
  - **exp12:** input staging and results transfer. Both were checked, with no effect.
  - **exp14:** M1-ctrl and M1-iter re-trained after the first account became inaccessible. The weights differ, and the deviation was reviewed by Codex before any ranking. Fidelity check: 0 of 400 hit@5 changes. The router fit is not bit-reproducible across scikit-learn environments (+0.027 vs +0.023), and the non-inferiority result flips. Version pinning was missing from the protocol.
  - **exp16:** A1 was loaded from either Drive, with the sha check unchanged; no effect.
  - **exp15:** the ≥3-hop subgroup was chosen from exp13 recall on the same questions. This is declared, and it is the weakest inferential point of the F1 claim; say so in the threats.

### 12. Moderate: missing threats to validity (`06_plan.tex:32-39`)
Add the following:
- **One training seed per delta in the grafting analyses.** Seeds were run for M1-iter and M1-ctrl, but the graft used seed 0 only. Rank-2 recovery across seeds is untested (HYPOTHESES H9: "one seed per delta").
- **Controls are single draws on tuning data** (it is stated in `05_results.tex:110`, but not in the threats).
- **Truncation:** 28% of 2WikiMultiHopQA iterative queries and 6% of HotpotQA queries hit the 512-token limit. In exp14 the primary is −0.011 when the query is truncated.
- **One reader** (`gpt-5.4-mini`), a closed API that can change; answer F1 depends on it. W3 plans a second reader, so cross-reference it.
- **Distribution shift in exp14:** it was evaluated on unused *training* questions (about 74% two-hop), not on dev.
- **Construct validity:** full-support recall depends on the gold annotations. Iteration could find valid alternative evidence that is not annotated. HotpotQA and MuSiQue annotations are partial.
- **Sequential, outcome-informed design.** exp14 was designed after exp13 failed, and the four-hop focus came from an exploratory exp12 reading. This is legitimate under pre-registration, but name it as a garden-of-forking-paths risk that the locked set addresses.
- **AI-assisted analysis.** The reviewer of each protocol was itself an LLM (Codex), so "independent" (`04_method.tex:63`) is generous. Name the model, and say that the reviews are in the repository (`experiments/reviews/`) and that the advisor approved each protocol, if true.

### 13. Minor: the protocol section does not say who checks the pre-registration
- `04_method.tex:63`. A committee will accept "committed before scoring" only if it can be verified.
- **Fix.** Add: "Registration commits are listed per experiment in Appendix~\ref{app:index}; any reader can check with `git log` that each protocol predates its results." Add the commit column in the index (finding 14).

---

## D. Chapter 6: follow-up experiments

### 14. Major: the W items are one-paragraph intentions without a question, design, decision rule or expected outcome
- `06_plan.tex:24-30`. The locked-set confirmation appears only in the schedule (`:90`).
- W2's suggested feature, "a cheap first iterative step", costs about 19 passes. The rule must count those passes.
- **Fix:** replace the `description` list with an intro sentence plus Table `tab:followups`. Each row becomes a future PROTOCOL.md; the decision rules below are proposals for the author to adjust before registration.

```latex
\section{Follow-up experiments}
\label{sec:followups}

Each follow-up will be pre-registered with the protocol of Section~\ref{sec:protocol} before any evaluation data are scored. Table~\ref{tab:followups} states its question, design, decision rule and the outcome expected from the current evidence. Thresholds are provisional and will be fixed in each protocol together with a power calculation.

\begin{table}[htbp]
\centering
\caption{Follow-up experiments. ``Base'' is the one-shot model run iteratively; recovery is Equation~\ref{eq:recovery}.}
\label{tab:followups}
\scriptsize
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.4cm}}
\toprule
Item & Question & Design & Decision rule & Expected outcome \\
\midrule
W1 Direct training &
Can the rank-2 change be learned directly as a LoRA adapter instead of extracted by SVD? &
On M1-ctrl, train LoRA adapters of rank 1, 2, 4 on M1-iter's 18,101 conditioned pairs (same order, schedule, 3 seeds); compare with the extracted rank-2 graft and with full fine-tuning on MuSiQue, 2Wiki and HotpotQA. Principal angles between the LoRA and extracted subspaces (descriptive). &
Primary: rank-2 LoRA, run iteratively, is non-inferior to the extracted rank-2 graft (margin 0.02 full-support recall, one-sided 95\% bound). Secondary: recovery $\ge 0.8$ on each dataset. &
Likely non-inferior; LoRA may use different directions than the SVD \cite{shuttleworth2025illusion}. \\
\addlinespace
W2 Better router &
Can a better benefit predictor close the gap to the oracle (0.94 recall@5, 27 passes)? &
Same splits as exp14 plus long-chain data. Candidate predictors: score-distribution features, a small classifier on the one-shot model's [CLS] states, and features from one iterative step (its passes counted). Cost is total passes including the predictor. &
Primary: the new cascade beats the exp14 benefit cascade by $\ge 0.01$ recall@5 at no more passes (McNemar). Secondary: AUC $\ge 0.70$. &
Small gain; headroom is large but benefit is noisy per question. \\
\addlinespace
W3 Answer quality &
Do the adapter and router preserve the answer-F1 gain, and with a second reader? &
Repeat exp15 with M1-ctrl+adapter and the cascade, readers \texttt{gpt-5.4-mini} and one open model. &
Primary: F1 on $\ge 3$-hop questions, cascade vs one-shot, CI above 0. &
Gain retained, smaller with the graft. \\
\addlinespace
W4 Geometry &
Is the conditioning change also \emph{spectrally} low-rank, and are its directions stable across seeds and datasets? &
For each layer and dataset (3 datasets $\times$ 3 seeds): energy $E_r$, stable rank and principal angles between top-2 subspaces. Baselines: seed-to-seed deltas and Gaussian matrices of equal norm \cite{altaf2026noisebaseline}. Rank-0 graft (one-dimensional deltas only). &
Directions are ``shared'' if the mean cosine of the first principal angle across seeds (and across datasets) exceeds the 95th percentile of the seed-noise baseline. Rank-0 graft recovery $< 0.2$. &
Shared across seeds; partly shared across datasets; energy not much above noise. \\
\addlinespace
W5 Transfer &
Does a change extracted on one dataset or reranker work on another? &
All exp16 models start from A1, so $\Delta_2$ from dataset $A$ can be grafted onto the one-shot model of dataset $B$ (6 ordered pairs); then onto A1 and A0 directly. Random and seed-noise controls on the target. &
Transfer if recovery $\ge 0.5$ with lower bound $> 0.25$ and above both controls, per pair. &
Transfer across datasets; weaker onto A0. \\
\addlinespace
Locked set &
Do the headline claims hold on data frozen now and never inspected? &
Freeze unused MuSiQue train IDs (sha committed) and unused 2Wiki/HotpotQA dev IDs; test at most three claims: rank-2 recovery, benefit cascade, long-chain gain. &
The original decision rules, Holm-corrected over the three claims. &
Rank-2 holds; long-chain test may be underpowered (about 200 four-hop questions). \\
\bottomrule
\end{tabularx}
\end{table}
```

- Keep the `tab:rq` mapping (`06_plan.tex:41-60`), but add "Locked set" to the RQ3 and RQ4 rows.
- In the schedule (`:86-90`), W3 depends on W1 and W2. Move it after them, or say that the first W3 run uses the extracted graft.

### 15. Minor: the "main novelty" sentence is in the plan, not in the contribution
- `06_plan.tex:29`: "Transfer of a behaviour ... would be the main novelty". This contradicts the abstract and introduction, where the main finding is the rank-2 structure.
- **Fix.** Say "would strengthen the main contribution".

---

## E. Appendix experiment index

### 16. Moderate: the index is written for the author, not for a reader outside the project
- `A_experiment_index.tex:9-30`. IDs are bare, "Question and outcome" mixes both in shorthand ("A1 − A0 is low-rank (rank 4: 80%)"), and there is no status, registration commit or deviation.
- **Fix:** one block per experiment, grouped by research question, using a `description` list or a longtable. Proposed format:

```latex
\chapter{Experiment Index}
\label{app:index}

Each experiment has a folder \texttt{experiments/expNN\_name/} in the project repository with a pre-registered protocol (\texttt{PROTOCOL.md}, committed before any evaluation data were scored) and a results report (\texttt{README.md}). Status: \textbf{C} confirmed by the pre-registered primary test; \textbf{N} the primary test failed; \textbf{E} exploratory or not testable. Experiments marked ``not used'' are kept for completeness.

\newcommand{\expentry}[6]{%
  \item[#1] \textbf{Question.} #2\\
  \textbf{Answer.} #3\\
  \textbf{Status:} #4. \textbf{Registered at:} \texttt{#5}. \textbf{In text:} #6.}

\section*{Token cost of a small selector (RQ1, RQ2)}
\begin{description}
  \expentry{Experiment 2}{How many prompt tokens does a fine-tuned 22.7M-parameter selector save on HotpotQA?}{68--77\% fewer tokens than sending all ten paragraphs, with no significant change in answer F1.}{C}{<commit>}{Section~\ref{sec:res_selector}}
  \expentry{Experiment 8}{Does the HotpotQA selector find all evidence for MuSiQue questions?}{Only for 68\% of questions within the top ten, so the planned stopping comparison could not be run.}{E (not testable)}{<commit>}{Section~\ref{sec:res_bottleneck}}
  % ...
\end{description}

\section*{Does conditioning help? (RQ3)}
% exp09, exp12 (N), exp13 (C four-hop; N hop router), exp14 (C; deviation: models re-trained), exp15 (C)

\section*{Structure of the weight change (RQ4)}
% exp06, exp11, exp12 secondary, exp16

\section*{Not used in the text}
% exp01, exp03, exp04, exp05, with one-line answers
```

Notes on the proposed index:
- Use plain-language questions, for example "Can the selector's skill be found in a few weights?" rather than "A1 − A0 is low-rank".
- Report the answer with the number and its unit.
- Mention the deviation inside the entry, for example for exp14: "Deviation: models re-trained after the original files became inaccessible; same recipe, identical test behaviour".
- The registration commits are in each README (for example exp12 `975371a`, exp14 `11296db`, exp16 `353fcd8`).

---

## F. Smaller points

17. **Minor**, `04_method.tex:58`: "where $R$ is full-support recall@5" contradicts exp16 (recall@(|gold|+1)). Fixed by Definition 1 in finding 7.
18. **Minor**, `02_background.tex:32`: "every row of the term is the same pattern, only scaled" is correct (rows of $u v^\top$ are multiples of $v^\top$). Add "and every column a multiple of $u$" so that both factors are interpreted.
19. **Minor**, `05_results.tex:42`: "Recovery was 74% at rank 1 and 95% at rank 2 ... (97% at ranks 4 and 8)". These are dev300 values for the confounded delta. Label them "tuning set" in the sentence, as is done later for the test value.
20. **Minor**, `01_introduction.tex:25`: the hypothesis says the change matters "mainly for long reasoning chains". exp14 shows that hop count does not predict benefit (finding 4, unclaimed result 2). Add half a sentence: "long chains are where the benefit concentrates on average, but per question it is better predicted from the one-shot scores than from the hop count".
