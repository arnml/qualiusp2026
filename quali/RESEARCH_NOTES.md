# Research notes: qualification text (PPGEE/Poli-USP), structure, writing, Overleaf

Compiled 2026-11-15. Legend: **[V]** = I opened the source and checked it. **[P]** = common practice or my inference, not a rule. **[?]** = could not verify.
Caveat: USP's legislation site (leginf.usp.br) returned HTTP 503 during this research, so I could **not** read the current PPGEE regulation (Res. CoPGr 7658/2019). Confirm the open points with the PPGEE secretariat or the advisor.

## 1. PPGEE / Poli-USP rules for the master's qualification

### 1a. Rule hierarchy (newest prevails)
| Level | Document | Status |
|---|---|---|
| USP | Regimento de Pós-Graduação, Res. 7493/2018 (amended by 8359/2022 and 8776/2025) | [V] text read via [FM-USP copy](https://fm.usp.br/posgrad/conteudo/Upload/regimento_de_pos_graduacao_atual.pdf) (leginf snapshot dated 22/08/2024) |
| Program | PPGEE Regulation, Res. CoPGr 7658/2019 ([leginf](http://www.leginf.usp.br/?resolucao=resolucao-copgr-no-7658-de-20-de-maio-de-2019)) | [?] listed as current on [PPGEE Normas](https://ppgee.poli.usp.br/en/?p=120); page was down |
| Program | [PPGEE FAQ](https://ppgee.poli.usp.br/pb/faq/) | [V] undated |
| Program (old) | [Normas PPGEE 2011](https://ppgee.poli.usp.br/wp-content/uploads/sites/92/2014/04/Normas-PPGEE-2011.pdf), Sec. VIII | [V] superseded, but the only text found that lists required contents |

### 1b. What is verified
| Topic | Rule | Source |
|---|---|---|
| Mandatory? | The USP Regimento makes it optional for the master's and leaves this to each program (Art. 71). PPGEE makes it **mandatory** for the master's. | [V] Regimento Art. 71; [V] FAQ |
| Purpose | To assess "a maturidade do aluno na área" (Regimento Art. 72). For the master's, PPGEE assesses the **relevance of the topic, the feasibility and schedule of execution, and the student's ability** to carry out the project. | [V] Regimento; [V] FAQ |
| Format | A written **monograph** plus an **oral presentation** of the research project. The committee also looks at the academic transcript. | [V] FAQ |
| Required contents of the text (2011 norm) | (a) objective; (b) justification; (c) methodology; (d) **cronograma**; (e) **complete literature review**; (f) **structured outline (sumário) of the dissertation**. Due **at least 15 days before** the exam. | [V] Normas 2011, VIII.3 (old; treat as the expected baseline) |
| Prerequisite | At least 70% of the course credits completed. | [V] FAQ; [V] Normas 2011 |
| Deadline to register | **18 months** after first enrollment as a regular student (students admitted after 2013). The 2011 norm said 21 months. The Regimento caps registration at 50% of the program's maximum deposit deadline (PPGEE: 36 months, so 18 months is consistent). Scholarship holders may have shorter deadlines in their Termo de Compromisso. | [V] FAQ; [V] Regimento Art. 72 §2 |
| Exam date | Within **90 days** of registration (Regimento). The old 2011 norm said 120 days, but the Regimento prevails. | [V] Regimento Art. 72 §2 |
| In person / remote | In person or remote. At least one examiner who is a program faculty member must be physically present at USP. | [V] Regimento Art. 72 §3 |
| Committee | Three members with at least a doctorate, **advisor included**, plus one alternate. The advisor proposes the committee at registration and the CCP approves it. A majority vote decides. | [V] FAQ; [V] Regimento Art. 73–74 |
| Outcome | Pass or fail, with no grade. The committee may recommend moving to the direct doctorate. One retake is allowed; failing it again means dismissal. | [V] Regimento Art. 73; [V] FAQ |
| Total duration | The master's ends at deposit: PPGEE **36 months** (USP maximum 48). | [V] FAQ; [V] Regimento Art. 43 |
| Language | Dissertations may be written and defended in **Portuguese, English or another language**. All must include **title, abstract and keywords in both PT and EN**. | [V] Regimento Art. 84; [V] FAQ |
| Language of the qualification text | No explicit rule found. Since the dissertation itself may be in English, an English text is consistent. | [P] Include a Portuguese *resumo* anyway, and confirm with the advisor. |
| Length / page limit | None found for PPGEE. | [?] Other USP programs cap the text at about 20 pages ([EESC-CEM example](https://pgr-cem.eesc.usp.br/en/qualifying-exam/)). |
| Oral duration | None found for PPGEE. Another Poli program (PPGET) uses 20–30 min plus questioning ([PPGET](https://sites.usp.br/ppget/informacoes/informacoes-alunos-regulares/)). | [P] |
| Forms | "Ata da banca julgadora sobre exame de qualificação" is on the [Poli graduate services page](https://www.poli.usp.br/en/134598pagina-de-atendimento-de-pos-graduacao/). | [V] |

### 1c. Formatting (Poli library / USP)
- The current guide is *Diretrizes para apresentação de dissertações e teses da USP: parte I (ABNT)*, **5th ed., 2024**, ABCD/USP, DOI 10.11606/9786598386221 ([book page](https://livrosabertos.abcd.usp.br/portaldelivrosUSP/catalog/book/1353)). [V] I verified the metadata only; I did not read the full text.
- The Poli library's Oct 2025 slides ([PDF](https://www.poli.usp.br/wp-content/uploads/2025/12/escrita-e-publicacao-cientifica.pdf)) [V] say Poli follows the USP Diretrizes, with these complements:
  - Deposit is digital only (since 08/03/2021).
  - There is no errata; corrections go only through the "versão corrigida".
  - **Illustrations carry their identification/caption at the bottom**, followed by the source. Tables follow the Diretrizes. Check what `politex-xico.cls` does here.
  - The cover page must say "Versão original" or "Versão corrigida", and the catalog card comes from the library.
- These apply to the final dissertation. For the qualification text they are good practice, but no rule requiring them was found [P].
- References follow ABNT NBR 6023 / NBR 10520 (same slides) [V]. This fits the `abntex2cite` setup already in the repo.

### 1d. Examples of Poli qualification texts
- No public Poli/PPGEE qualification texts were found; programs don't publish them [?].
- [P] Ask the advisor's lab (PCS) for 1–2 recent ones. Use final PCS dissertations in the USP digital library (teses.usp.br) as style references.

## 2. Recommended structure for a CS/ML qualification text

What examiners check, according to the guides, maps onto PPGEE's criteria: (1) **Is the problem relevant and well scoped for a master's?** (2) **Is the plan feasible in the time left?** (3) **Can the student do it, as shown by command of the literature and preliminary results?**

| Guide | Key requirements |
|---|---|
| [U. Manitoba CS, "Guidelines for Writing a Successful MSc Thesis Proposal"](https://www-elec.inaoep.mx/~rogerio/ProposalGuidelines.pdf) (2004, INAOE mirror) [V] | Problem statement in 1–3 sentences near the start. Representative (not exhaustive) related work. A method convincing enough that it will work. **Exactly how success will be evaluated.** Expected results and why they matter. **Milestone timeline.** Resources. 10–20 pages. |
| [Boise State Computing PhD handbook, Dissertation Proposal](https://sites.google.com/boisestate.edu/computingphdstudenthandbook/dissertation-proposal) [V] | Objective, literature review, motivation and **expected contributions**, **preliminary results**, methodology **with success metrics**, **timeline for publications and chapters**. Maximum 15 pages. |
| PPGEE FAQ + 2011 norm (above) [V] | Objective, justification, methodology, cronograma, complete literature review, **dissertation outline**. |

**Suggested chapter list** for this project [P]. It is a synthesis of the sources above and matches the existing `chapters/` files.
1. **Introduction**: motivation (cost of multi-hop RAG); problem statement; research questions/hypotheses; **expected contributions** (a numbered list, each tied to evidence or to planned work); text organization.
2. **Background**: RAG, multi-hop QA, cross-encoders/rerankers, adaptive retrieval and stopping, low-rank weight analysis.
3. **Related work**: grouped by approach, ending with a positioning table or gap statement.
4. **Proposed method / methodology**: selector, stopping rule, rank-2 analysis; datasets, metrics, baselines, **statistical protocol** (pre-registration, CIs, tests).
5. **Preliminary results**: what has been done (incl. the BRACIS 2026 paper); confirmed vs exploratory results kept clearly apart; negative results stated.
6. **Work plan**: remaining research questions, then experiments, then expected outputs; **cronograma** (Gantt-style table by month up to the deposit date, about 36 months after enrollment); risks and fallbacks; target venues.
7. **Proposed dissertation outline** (*sumário estruturado*). This is required by the 2011 norm, so keep it.
- Pre-textual: cover, abstract (EN) **+ resumo (PT)**, keywords in both languages, lists.
- Post-textual: references; appendices (experiment index).

## 3. Academic writing practices

**Claims vs evidence, hedging**
- Each claim gets a type: *confirmed* (pre-registered primary test), *exploratory*, or *conjecture*. The verb's strength should match: "shows" for confirmed, "suggests" for exploratory, "we hypothesize" for conjecture [P].
- Hedging devices to use: distancing ("it is thought that"), cautious explanations (modal verbs), and avoiding over-generalization ("in general"). See [Manchester Academic Phrasebank: Being cautious](https://www.phrasebank.manchester.ac.uk/using-cautious-language/) [V].
- Put the contributions up front. See Simon Peyton Jones, [How to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) [V: page exists; slides not read].

**Reporting statistics**
- [ASA Statement on p-values (2016)](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) [V]:
  - A p-value does not measure effect size or importance.
  - Don't decide on a threshold alone.
  - Report fully and transparently.
- So report the **effect size + 95% CI + p**, plus n, the test used, and the unit of resampling. Example: "+0.055 F1 [0.007, 0.102], paired bootstrap, n=…, p=0.035".
- [NeurIPS paper checklist, item 7](https://neurips.cc/public/guides/PaperChecklist) [V]:
  - State what variability the error bars capture and how they were computed.
  - Say whether a bar is the SD or the SE.
  - 1-sigma bars are acceptable if labeled; 2-sigma is preferred when normality is not checked.
- Say which analyses were pre-registered. Correct for multiple comparisons, or label results as exploratory [P].

**Plagiarism / self-plagiarism**
- No central USP self-plagiarism policy was found [?]. Self-plagiarism is broadly treated as reusing one's earlier text without citing it [P].
- At deposit, Poli requires an **originality statement / plagiarism report** ([Poli page](https://www.poli.usp.br/en/134598pagina-de-atendimento-de-pos-graduacao/)) [V].
- USP licenses **Turnitin (OriginalityCheck)** for graduate faculty through ABCD ([2025 tutorial](https://www.abcd.usp.br/wp-content/uploads/2025/06/TUTORIAL-TURNITIN-PARA-PROFESSOR-2025.pdf)) [V]. The advisor can run the text through it.
- **Reusing the BRACIS 2026 paper** (Springer LNCS 17106, per `bibli.bib`). The standard Springer Nature proceedings Licence to Publish lets the author "reuse the Version of Record or any part in a thesis written by the same Author" and deposit it in the institutional repository, with the acknowledgement "Reproduced with permission from Springer Nature" ([LTP template](https://safecomp2026.webs.upv.es/wp-content/uploads/2026/01/SNCS_ProceedingsPaper_LTP_ST_SN_Switzerland.pdf)) [V template; **check the copy you actually signed**].
  - In practice [P]: cite the paper at the start of each chapter that reuses it, with a footnote such as "Parts of this chapter are adapted from [Lavado et al. 2027]…". Prefer rewriting or extending over copying. Don't reuse co-authors' figures without credit.

**Generative-AI disclosure**
- **USP**: no central rule found. The FFCLRP-USP graduate committee (Jan 2026) explicitly cites "a atual ausência de diretrizes formais específicas para utilização de inteligência artificial na Universidade de São Paulo" ([FFCLRP declaration + guidelines](https://dcm.ffclrp.usp.br/profmat/arquivos/formularios/695e6716bb198_DeclaracaoUsoInteligenciaArtificial.pdf)) [V].
  - That unit requires a signed declaration: the tools used, how they were used (language revision, translation, structure, etc.), and a statement of critical review.
  - It forbids AI-generated data or results and unreviewed, undeclared text.
  - It is not binding on Poli, but it is a good model [P].
- **CAPES**: no specific national rule on AI in theses was found [?]. (CAPES IN 1/2026 is about data interoperability, not AI.)
- Recommendation [P]: add a short "Use of AI tools" note to the front matter or acknowledgements, modeled on the FFCLRP form, saying which tools were used, for what, and that the author reviewed and takes responsibility for all content. Confirm with the advisor.

## 4. Overleaf + GitHub in 2026

| Fact | Detail | Source |
|---|---|---|
| Git integration and GitHub sync | **Premium only** on Overleaf Cloud. | [V] [Git/GitHub docs](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization), [premium features](https://docs.overleaf.com/getting-started/free-and-premium-plans/premium-features) |
| Premium is per project | If the **owner** is premium, free collaborators can sync, push and pull. | [V] premium features |
| GitHub sync behavior | Manual push/pull (not automatic). Only the owner can link. You can import an existing repo as a *new* project, but you can't link an existing project to an existing repo. No submodules or LFS. Sync stops above 7 MB of editable content. | [V] [GitHub sync docs](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization) |
| USP license | **None.** Overleaf states USP "does not currently have an institutional partnership with Overleaf". A free account with a USP email is still possible. | [V] [overleaf.com/edu/usp](https://www.overleaf.com/edu/usp) |
| Free plan limits | **10 s compile timeout** (premium: 240 s); **1 collaborator** (viewers unlimited); 24 h history; no track changes; 7 MB editable; 2,000 files; 50 MB per upload. | [V] [plan limits](https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits), premium features |
| Zip import | New project → "Existing project (.zip)". Up to 180 files and 50 MB per upload. | [V] [docs](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project.md) |

**Risk** [P]: a 10 s timeout is tight for an abnTeX2 thesis with TikZ figures (`figures/*.tex`) and bibliography passes. Mitigations:
- Compile with draft mode or `\includeonly` while editing.
- Externalize or pre-render TikZ figures to PDF.
- Keep the local or CI build as the authoritative one.

**Recommended (free) workflow: GitHub is the source of truth; Overleaf is a viewer and editor.** The repo is ~300 KB, well within limits.
1. Keep `quali/` in the GitHub repo and build locally with `latexmk -pdf main.tex` (TeX Live/MiKTeX + VS Code LaTeX Workshop) [P].
2. Optional CI: a GitHub Action with [`xu-cheng/latex-action@v4`](https://github.com/xu-cheng/latex-action) (`root_file: main.tex`; full TeX Live; uses latexmk) builds the PDF on every push and stores it as an artifact [V]. This gives a no-timeout reference PDF.
3. For Overleaf: zip the `quali/` folder and create the project via New project → Existing project (.zip) [V]. Invite the advisor as the **single** free collaborator, or share a read-only link (viewers are unlimited) [V].
4. Round-trip edits made in Overleaf: Menu → Download → Source (zip), unzip over `quali/`, review with `git diff`, then commit [P]. Rule: only one side edits at a time, to avoid manual merges.

**Fallback / upgrade**: if Overleaf will be used heavily, **one premium owner (e.g., Overleaf Student plan) is enough**. Then use New project → GitHub repo (import), and manual push/pull from Integrations → GitHub. Or use the Overleaf git remote (`git clone` of the project) and add GitHub as a second remote [V: features exist; per-project premium]. Alternatively, an advisor with a premium account can own the project, and the student syncs it for free [V].
