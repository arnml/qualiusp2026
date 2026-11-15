# Published versions of bibliography entries

Date: 2026-11-15. Master: `bibli.bib` (identical copy in `quali/bibli.bib`). 103 entries, 50 cited in `quali/chapters/*.tex` (bold below).

## Summary

- Preprint (or arXiv-note-only) entries converted to the published version: **35** (incl. 3 with an extra fix noted in the table)
- Still preprints (no published version found on official sources; normalized to `@misc` + arXiv fields, `note = {arXiv preprint}`): **18**
- Errors fixed: **3** (RECOMP published title; Gromov et al. year 2025 instead of 2024; MICE author list 6 -> 4)
- Already-published entries only completed (pages / official URL, venue unchanged): **10**
- Other entries unchanged: **39**
- No entry deleted or reordered; same keys.

## Notes and caveats

- Official sources used: ACL Anthology (`.bib` files), proceedings.neurips.cc (citation meta tags), proceedings.mlr.press, ICLR proceedings page, ICML 2026 virtual site, ACM DL / Springer / Elsevier / MIT Press via DOI resolver (doi.org BibTeX) and Crossref, OpenReview search API (venue field) for ICLR/COLM records.
- OpenReview forum pages sit behind a browser check, so ICLR 2018-2023 entries (li2018intrinsic, ilharco2023taskarithmetic, asai2020pathretriever, hu2022lora) were left unchanged; their venue rests on the arXiv comment ("Published in ICLR 2018", "ICLR 2023", "ICLR 2020") and, for hu2022lora, the existing OpenReview URL. They were not re-verified on an official page.
- **liu2026unifieddefinitionhallucinationits**: accepted to ICML 2026 as a *position paper poster* under the title "Position: We Need A Unified Definition of Hallucination (It's The World Model, Stupid!)" (icml.cc). No PMLR page was found, so the entry has no pages/volume. Title differs from the arXiv title.
- **xu2024recomp**: ICLR 2024 title is "RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation" (OpenReview venue record, ICLR 2024 poster).
- turner2025modelorganisms, soligo2025convergent: only workshop versions exist (ICML R2-FM / NeurIPS Mech Interp workshops), not archival, so kept as preprints.
- altaf2026noisebaseline: "Under review at TMLR" per arXiv comments, so a preprint.
- wang2023filco, pradeep2023rankzephyr, morris2026tinylora, ye2026rlvrlowrank: no venue found.
- lavado2027noisegate: Springer print year is 2027 (Crossref), online 2026-10-02; year kept at 2027, pages 454-468 added.
- michel2019sixteenheads: NeurIPS 2019 page gives no page numbers. khattab2021baleen: no DOI on the page.
- liang2025draganddrop: nickname "Atlas" in the NeurIPS author string dropped.
- sajjad2023droppinglayers: Elsevier vol. 77 (Jan 2023), article number 101429 (arXiv comment said 2022).
- Entries not checked for a newer version because they were never preprints: Guu (PMLR), Lewis (NeurIPS), Jin (Findings ACL), Guo (Nature), Smithies, Hu (kept), all ACL Anthology / TACL / PNAS / Annals items.

## Table

| key | before | after | official source | status |
|---|---|---|---|---|
| **pmlr-v119-guu20a** (cited) | 37th Int. Conf. Machine Learning 2020 | 37th Int. Conf. Machine Learning 2020 | https://proceedings.mlr.press/v119/guu20a.html | unchanged |
| **3495724.3496517** (cited) | 34th Int. Conf. Neural Information Processing Systems 2020 | 34th Int. Conf. Neural Information Processing Systems 2020 | - | unchanged |
| asai2024reliableadaptableattributablelanguage | arXiv:2403.03187 2024 | arXiv:2403.03187 2024 | https://arxiv.org/abs/2403.03187 | still preprint |
| jin-etal-2025-rag | Findings of the Association for Computational Linguistics: A 2025 | Findings of the Association for Computational Linguistics: A 2025 | https://aclanthology.org/2025.findings-acl.877/ | unchanged |
| Smithies_Berman_Taiuru_Spychal_Moore_2026 |   |   | - | unchanged |
| **reimers2019sentence** (cited) | arXiv preprint arXiv:1908.10084 2019 | 2019 Conference on Empirical Methods in Natural Language Pro 2019 | https://aclanthology.org/D19-1410/ | updated |
| **liu2026unifieddefinitionhallucinationits** (cited) | arXiv:2512.21577 2026 | 43rd Int. Conf. Machine Learning (ICML) 2026 | https://icml.cc/virtual/2026/poster/67063 | updated |
| Guo_2025 | Nature 2025 | Nature 2025 | http://dx.doi.org/10.1038/s41586-025-09422-z | unchanged |
| lambert2025tulu3pushingfrontiers | arXiv:2411.15124 2025 | Second Conference on Language Modeling (COLM) 2025 | https://openreview.net/forum?id=i1uGbfHHpH | updated |
| **lavado2027noisegate** (cited) | Intelligent Systems. BRACIS 2026 2027 | Intelligent Systems. BRACIS 2026 2027 | https://doi.org/10.1007/978-3-032-39892-5_30 | unchanged; completed (pages) |
| takehi2026weightspace | Blog post, MIT CSAIL 2026 | Blog post, MIT CSAIL 2026 | https://rikiyat.github.io/blog/weight-space-programming/ | unchanged |
| ahmed2026weightspaceaudit | arXiv:2611.08070 2026 | arXiv:2611.08070 2026 | https://arxiv.org/abs/2611.08070 | still preprint |
| li2026tokeneconomizer | arXiv:2610.22732 2026 | arXiv:2610.22732 2026 | https://arxiv.org/abs/2610.22732 | still preprint |
| charakorn2025texttolora | 42nd Int. Conf. Machine Learning (ICML) 2025 | 42nd Int. Conf. Machine Learning 2025 | https://proceedings.mlr.press/v267/charakorn25a.html | unchanged venue; completed (PMLR vol/pages/URL) |
| liang2025draganddrop | Advances in Neural Information Processing Systems (NeurIPS) 2025 | Advances in Neural Information Processing Systems 2026 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/e15b6539e6d61aab5dee4593222ba40e-Abstract-Conference.html | updated |
| jeong2025ecorag | Findings of the Association for Computational Linguistics: A 2025 | Findings of the Association for Computational Linguistics: A 2025 | https://aclanthology.org/2025.findings-acl.1365/ | updated |
| zhang2025utilityselector | arXiv:2507.19102 2025 | 2025 Annual International ACM SIGIR Conference on Research a 2025 | https://doi.org/10.1145/3767695.3769496 | updated |
| zhu2025contextpicker | arXiv:2512.14465 2025 | arXiv:2512.14465 2025 | https://arxiv.org/abs/2512.14465 | still preprint |
| li2025brief | Findings of the Association for Computational Linguistics: N 2025 | Findings of the Association for Computational Linguistics: N 2025 | https://aclanthology.org/2025.findings-naacl.301/ | updated |
| zhou2026dynakrag | arXiv:2607.06507 2026 | arXiv:2607.06507 2026 | https://arxiv.org/abs/2607.06507 | still preprint |
| **chirkova2025provence** (cited) | Int. Conf. Learning Representations (ICLR) 2025 | Int. Conf. Learning Representations (ICLR) 2025 | https://openreview.net/forum?id=TDy5Ih78b4 | unchanged venue; added official URL |
| **xu2024recomp** (cited) | Int. Conf. Learning Representations (ICLR) 2024 | Int. Conf. Learning Representations (ICLR) 2024 | https://openreview.net/forum?id=mlJLVigNHp | fixed error (published title is "...with Context Compression and Selective Augmentation") |
| **jiang2024longllmlingua** (cited) | Proceedings of ACL 2024 | 62nd Annual Meeting of the Association for Computational Lin 2024 | https://aclanthology.org/2024.acl-long.91/ | updated |
| cheng2024xrag | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/c5cf13bfd3762821ef7607e63ee90075-Abstract-Conference.html | updated |
| **xiong2021mdr** (cited) | Int. Conf. Learning Representations (ICLR) 2021 | Int. Conf. Learning Representations (ICLR) 2021 | https://iclr.cc/virtual/2021/poster/2806 | unchanged venue; added official ICLR URL |
| **zhang2024beamretrieval** (cited) | Proceedings of NAACL 2024 | 2024 Conference of the North American Chapter of the Associa 2024 | https://aclanthology.org/2024.naacl-long.96/ | updated |
| **taguchi2025adaptivek** (cited) | Proceedings of EMNLP 2025 | 2025 Conference on Empirical Methods in Natural Language Pro 2025 | https://aclanthology.org/2025.emnlp-main.1017/ | updated |
| lee2025setr | Proceedings of ACL 2025 | 63rd Annual Meeting of the Association for Computational Lin 2025 | https://aclanthology.org/2025.acl-long.861/ | updated |
| alain2016probes | arXiv:1610.01644 2016 | arXiv:1610.01644 2016 | https://arxiv.org/abs/1610.01644 | still preprint |
| hewitt2019controltasks | Proceedings of EMNLP-IJCNLP 2019 | 2019 Conference on Empirical Methods in Natural Language Pro 2019 | https://aclanthology.org/D19-1275/ | updated |
| belinkov2022probing | Computational Linguistics 2022 | Computational Linguistics 2022 | https://doi.org/10.1162/coli_a_00422 | updated |
| marks2023geometrytruth | arXiv:2310.06824 2023 | First Conference on Language Modeling (COLM) 2024 | https://openreview.net/forum?id=aajyHYjjsk | updated |
| arditi2024refusal | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html | updated |
| michel2019sixteenheads | Advances in Neural Information Processing Systems (NeurIPS) 2019 | Advances in Neural Information Processing Systems 2019 | https://proceedings.neurips.cc/paper_files/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html | updated |
| voita2019heads | Proceedings of ACL 2019 | 57th Annual Meeting of the Association for Computational Lin 2019 | https://aclanthology.org/P19-1580/ | updated |
| sun2024wanda | Int. Conf. Learning Representations (ICLR) 2024 | Int. Conf. Learning Representations (ICLR) 2024 | https://openreview.net/forum?id=PxoFut3dWW | unchanged venue; added OpenReview URL |
| sajjad2023droppinglayers | Computer Speech \& Language 2023 | Computer Speech \& Language 2023 | https://doi.org/10.1016/j.csl.2022.101429 | updated |
| gromov2024deeperlayers | arXiv:2403.17887 2024 | Int. Conf. Learning Representations (ICLR) 2025 | https://openreview.net/forum?id=ngmEcEer8a | updated + fixed year 2024->2025 (ICLR 2025) |
| **panigrahi2023skilllocalization** (cited) | 40th Int. Conf. Machine Learning (ICML) 2023 | 40th Int. Conf. Machine Learning (ICML) 2023 | - | unchanged |
| yin2024lofit | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/122ea6470232ee5e79a2649243348005-Abstract-Conference.html | updated |
| hase2023localization | Advances in Neural Information Processing Systems (NeurIPS) 2023 | Advances in Neural Information Processing Systems 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html | updated |
| wu2024reft | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html | updated |
| vast2026mice | arXiv:2602.16299 2026 | 2026 Conference on Empirical Methods in Natural Language Pro 2026 | https://aclanthology.org/2026.emnlp-main.1736/ | updated + fixed author list (6 -> 4, per Anthology) |
| java2026frugalrag | Int. Conf. Learning Representations (ICLR) 2026 | Int. Conf. Learning Representations (ICLR) 2026 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/ede6f43d254731152970009c172d5561-Abstract-Conference.html | unchanged venue; completed (official URL, title casing as published) |
| **vasani2025phlora** (cited) | Findings of IJCNLP-AACL 2025 2025 | Findings of IJCNLP-AACL 2025 2025 | - | unchanged |
| **li2018intrinsic** (cited) | Int. Conf. Learning Representations (ICLR) 2018 | Int. Conf. Learning Representations (ICLR) 2018 | https://arxiv.org/abs/1804.08838 | unchanged |
| **aghajanyan2021intrinsic** (cited) | 59th Annual Meeting of the Association for Computational Lin 2021 | 59th Annual Meeting of the Association for Computational Lin 2021 | https://aclanthology.org/2021.acl-long.568/ | unchanged |
| **hu2022lora** (cited) | Int. Conf. Learning Representations (ICLR) 2022 | Int. Conf. Learning Representations (ICLR) 2022 | https://openreview.net/forum?id=nZeVKeeFYf9 | unchanged |
| **ilharco2023taskarithmetic** (cited) | Int. Conf. Learning Representations (ICLR) 2023 | Int. Conf. Learning Representations (ICLR) 2023 | https://arxiv.org/abs/2212.04089 | unchanged |
| **yadav2023ties** (cited) | Advances in Neural Information Processing Systems (NeurIPS) 2023 | Advances in Neural Information Processing Systems 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html | updated |
| liu2024bitdelta | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/187d94b3c93343f0e925b5cf729eadd5-Abstract-Conference.html | updated |
| ping2024deltacome | Advances in Neural Information Processing Systems (NeurIPS) 2024 | Advances in Neural Information Processing Systems 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/37664246a1e07e212ddacea6e5a523f2-Abstract-Conference.html | updated |
| **biderman2024loralearnsless** (cited) | Transactions on Machine Learning Research (TMLR) 2024 | Transactions on Machine Learning Research (TMLR) 2024 | https://openreview.net/forum?id=aloEru2qCG | unchanged |
| **shuttleworth2025illusion** (cited) | Advances in Neural Information Processing Systems (NeurIPS) 2025 | Advances in Neural Information Processing Systems 2026 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html | updated |
| mukherjee2025rlsubnetworks | Advances in Neural Information Processing Systems (NeurIPS) 2025 | Advances in Neural Information Processing Systems 2026 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/bf235a1d6780afd979f2f81676f43413-Abstract-Conference.html | updated |
| **morris2026tinylora** (cited) | arXiv:2602.04118 2026 | arXiv:2602.04118 2026 | https://arxiv.org/abs/2602.04118 | still preprint |
| **ye2026rlvrlowrank** (cited) | arXiv:2605.06523 2026 | arXiv:2605.06523 2026 | https://arxiv.org/abs/2605.06523 | still preprint |
| lu2025pathway | 2025 Conference on Empirical Methods in Natural Language Pro 2025 | 2025 Conference on Empirical Methods in Natural Language Pro 2025 | https://aclanthology.org/2025.emnlp-main.1297/ | unchanged |
| **perlman2026emergence** (cited) | arXiv:2608.23338 2026 | Findings of the Association for Computational Linguistics: E 2026 | https://aclanthology.org/2026.findings-emnlp.1948/ | updated |
| **braga2025taskarithmetic** (cited) | 48th International ACM SIGIR Conference on Research and Deve 2025 | 48th International ACM SIGIR Conference on Research and Deve 2025 | https://doi.org/10.1145/3726302.3730216 | unchanged; completed (pages) |
| xia2025crosslora | arXiv:2508.05232 2025 | arXiv:2508.05232 2025 | https://arxiv.org/abs/2508.05232 | still preprint |
| farhadzadeh2025lorax | Int. Conf. Learning Representations (ICLR) 2025 | Int. Conf. Learning Representations (ICLR) 2025 | https://openreview.net/forum?id=6cQ6cBqzV3 | unchanged venue; added OpenReview URL |
| asai2020pathretriever | Int. Conf. Learning Representations (ICLR) 2020 | Int. Conf. Learning Representations (ICLR) 2020 | https://arxiv.org/abs/1911.10470 | unchanged |
| **khattab2021baleen** (cited) | Advances in Neural Information Processing Systems (NeurIPS) 2021 | Advances in Neural Information Processing Systems 2021 | https://proceedings.neurips.cc/paper_files/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html | updated |
| **trivedi2023ircot** (cited) | 61st Annual Meeting of the Association for Computational Lin 2023 | 61st Annual Meeting of the Association for Computational Lin 2023 | https://aclanthology.org/2023.acl-long.557/ | unchanged |
| **zhuang2024efficientrag** (cited) | 2024 Conference on Empirical Methods in Natural Language Pro 2024 | 2024 Conference on Empirical Methods in Natural Language Pro 2024 | https://aclanthology.org/2024.emnlp-main.199/ | unchanged |
| erker2026grithopper | 19th Conference of the European Chapter of the Association f 2026 | 19th Conference of the European Chapter of the Association f 2026 | https://aclanthology.org/2026.eacl-long.5/ | unchanged |
| jain2025contextualutility | 14th International Joint Conference on Natural Language Proc 2025 | 14th International Joint Conference on Natural Language Proc 2025 | https://aclanthology.org/2025.ijcnlp-short.37/ | unchanged |
| meng2025dps | arXiv:2508.09497 2025 | arXiv:2508.09497 2025 | https://arxiv.org/abs/2508.09497 | still preprint |
| schlatt2025setencoder | Advances in Information Retrieval (ECIR 2025) 2025 | Advances in Information Retrieval (ECIR 2025) 2025 | https://doi.org/10.1007/978-3-031-88711-6_1 | unchanged; completed (pages) |
| **liu2024lostinthemiddle** (cited) | Transactions of the Association for Computational Linguistic 2024 | Transactions of the Association for Computational Linguistic 2024 | https://aclanthology.org/2024.tacl-1.9/ | unchanged |
| **wang2023filco** (cited) | arXiv:2311.08377 2023 | arXiv:2311.08377 2023 | https://arxiv.org/abs/2311.08377 | still preprint |
| **asai2024selfrag** (cited) | Int. Conf. Learning Representations (ICLR) 2024 | Int. Conf. Learning Representations (ICLR) 2024 | https://openreview.net/forum?id=hSyW5go0v8 | unchanged |
| **jeong2024adaptiverag** (cited) | 2024 Conference of the North American Chapter of the Associa 2024 | 2024 Conference of the North American Chapter of the Associa 2024 | https://aclanthology.org/2024.naacl-long.389/ | unchanged |
| **yoon2024compact** (cited) | 2024 Conference on Empirical Methods in Natural Language Pro 2024 | 2024 Conference on Empirical Methods in Natural Language Pro 2024 | https://aclanthology.org/2024.emnlp-main.1194/ | unchanged |
| **sun2025dynamicrag** (cited) | arXiv:2505.07233 2025 | Advances in Neural Information Processing Systems 2026 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/f69d2246a85e28c3e63dee3daf8d4af8-Abstract-Conference.html | updated |
| **sun2023rankgpt** (cited) | 2023 Conference on Empirical Methods in Natural Language Pro 2023 | 2023 Conference on Empirical Methods in Natural Language Pro 2023 | https://aclanthology.org/2023.emnlp-main.923/ | unchanged |
| **pradeep2023rankzephyr** (cited) | arXiv:2312.02724 2023 | arXiv:2312.02724 2023 | https://arxiv.org/abs/2312.02724 | still preprint |
| yoon2024listt5 | 62nd Annual Meeting of the Association for Computational Lin 2024 | 62nd Annual Meeting of the Association for Computational Lin 2024 | https://aclanthology.org/2024.acl-long.125/ | unchanged |
| **trivedi2022musique** (cited) | Transactions of the Association for Computational Linguistic 2022 | Transactions of the Association for Computational Linguistic 2022 | https://aclanthology.org/2022.tacl-1.31/ | unchanged |
| **card2020power** (cited) | 2020 Conference on Empirical Methods in Natural Language Pro 2020 | 2020 Conference on Empirical Methods in Natural Language Pro 2020 | https://aclanthology.org/2020.emnlp-main.745/ | unchanged |
| gargiulo2025tsv | IEEE/CVF Conference on Computer Vision and Pattern Recogniti 2025 | IEEE/CVF Conference on Computer Vision and Pattern Recogniti 2025 | https://openaccess.thecvf.com/content/CVPR2025/html/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.html | unchanged |
| huang2024chatvector | 62nd Annual Meeting of the Association for Computational Lin 2024 | 62nd Annual Meeting of the Association for Computational Lin 2024 | https://aclanthology.org/2024.acl-long.590/ | unchanged |
| cao2025paramdelta | Int. Conf. Learning Representations (ICLR) 2025 | Int. Conf. Learning Representations (ICLR) 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/78bca5cc621a0846cb1f8265e1927a2a-Abstract-Conference.html | unchanged |
| zbeeb2025reasoningvectors | arXiv:2509.01363 2025 | arXiv:2509.01363 2025 | https://arxiv.org/abs/2509.01363 | still preprint |
| tang2026past | arXiv:2601.11258 2026 | 64th Annual Meeting of the Association for Computational Lin 2026 | https://aclanthology.org/2026.acl-long.550/ | updated |
| rinaldi2026gradientsign | Int. Conf. Learning Representations (ICLR) 2026 | Int. Conf. Learning Representations (ICLR) 2026 | https://openreview.net/forum?id=CQnpxZNMvT | unchanged venue; added OpenReview URL |
| **turner2025modelorganisms** (cited) | arXiv:2506.11613 2025 | arXiv:2506.11613 2025 | https://arxiv.org/abs/2506.11613 | still preprint |
| **soligo2025convergent** (cited) | arXiv:2506.11618 2025 | arXiv:2506.11618 2025 | https://arxiv.org/abs/2506.11618 | still preprint |
| **altaf2026noisebaseline** (cited) | arXiv:2610.22438 2026 | arXiv:2610.22438 2026 | https://arxiv.org/abs/2610.22438 | still preprint |
| fleshman2024readaptir | arXiv:2406.14764 2024 | 48th International ACM SIGIR Conference on Research and Deve 2025 | https://doi.org/10.1145/3726302.3730240 | updated |
| kebir2026convmerging | arXiv:2607.08540 2026 | 49th International ACM SIGIR Conference on Research and Deve 2026 | https://doi.org/10.1145/3805712.3809939 | updated |
| **yang2018hotpotqa** (cited) | 2018 Conference on Empirical Methods in Natural Language Pro 2018 | 2018 Conference on Empirical Methods in Natural Language Pro 2018 | https://aclanthology.org/D18-1259/ | unchanged |
| **ho2020twowiki** (cited) | 28th Int. Conf. Computational Linguistics 2020 | 28th Int. Conf. Computational Linguistics 2020 | https://aclanthology.org/2020.coling-main.580/ | unchanged |
| **wang2020minilm** (cited) | Advances in Neural Information Processing Systems 2020 | Advances in Neural Information Processing Systems 2020 | https://proceedings.neurips.cc/paper_files/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf | unchanged |
| **nogueira2019passage** (cited) | arXiv:1901.04085 2019 | arXiv:1901.04085 2019 | https://arxiv.org/abs/1901.04085 | still preprint |
| **bajaj2016msmarco** (cited) | arXiv:1611.09268 2016 | arXiv:1611.09268 2016 | https://arxiv.org/abs/1611.09268 | still preprint |
| devlin2019bert | 2019 Conference of the North American Chapter of the Associa 2019 | 2019 Conference of the North American Chapter of the Associa 2019 | https://aclanthology.org/N19-1423/ | unchanged |
| **mcnemar1947** (cited) | Psychometrika 1947 | Psychometrika 1947 | - | unchanged |
| **efron1979bootstrap** (cited) | The Annals of Statistics 1979 | The Annals of Statistics 1979 | - | unchanged |
| **dror2018hitchhiker** (cited) | 56th Annual Meeting of the Association for Computational Lin 2018 | 56th Annual Meeting of the Association for Computational Lin 2018 | https://aclanthology.org/P18-1128/ | unchanged |
| **nosek2018preregistration** (cited) | National Academy of Sciences 2018 | National Academy of Sciences 2018 | - | unchanged |
| **vaswani2017attention** (cited) | Advances in Neural Information Processing Systems 2017 | Advances in Neural Information Processing Systems 2017 | https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf | unchanged |
