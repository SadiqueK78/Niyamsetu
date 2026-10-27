# Niyamsetu — evaluation results on the 25-GR corpus

Sources: `retrieval_scores_natural_20261027_074844.json`, `harness_natural_20261027_071414.json`. Figures: fig1_retrieval_by_config.png, fig2_hit1_by_language.png, fig3_hit1_by_tier.png, fig4_answer_accuracy.png, fig5_latency.png, fig6_corpus_chunks.png, fig7_unit_tests.png.

## Table 1. Corpus

| # | Document ID | Issue date | Type | Subject (short) | Characters | Chunks | Mean chunk (chars) |
|---|---|---|---|---|---|---|---|
| 1 | GAD_201811161110169207 | 2018-11-17 | GR | New Maharashtra Sadan (Delhi): room reservation & rates, consolidated | 61511 | 30 | 2115 |
| 2 | GAD_201812201818380207 | 2018-12-20 | GR | Government Communication (advertisement) Rules, 2018 | 117455 | 53 | 2252 |
| 3 | GAD_202102011814485107 | 2021-02-01 | Circular | Foreign tours of officers: norms & procedure (consolidated) | 55328 | 28 | 2015 |
| 4 | GAD_202102161256152607 | 2021-02-16 | GR | Posts identified for PwD in Lok Ayukta office (RPwD Act s.33) | 108248 | 65 | 1711 |
| 5 | GAD_202105201508210107 | 2021-05-20 | Circular | Section Officer provisional seniority list 1986–2016 | 63105 | 30 | 2161 |
| 6 | GAD_202108301543421007 | 2021-08-30 | Order | Ad hoc promotion: Clerk-Typist → Asst. Section Officer 2020-21 | 67233 | 30 | 2292 |
| 7 | GAD_202109011815562507 | 2021-08-30 | Order | Promotion: Asst. Section Officer → Section Officer 2020-21 | 86419 | 139 | 679 |
| 8 | GAD_202205041847101207 | 2022-05-04 | GR | Direct recruitment to Gr-B/C/D posts (ex-SSSB): consolidated guidelines | 56286 | 26 | 2216 |
| 9 | GAD_202305111444322707 | 2023-05-11 | Order | Inter-departmental transfers of Asst. Section Officers | 71094 | 31 | 2336 |
| 10 | GAD_202305111455014207 | 2023-05-11 | Order | Inter-departmental transfers of Clerk-Typists | 68358 | 30 | 2321 |
| 11 | GAD_202401091710039807 | 2024-01-09 | Memorandum | Clerk-Typist Exam 2021: waiting-list allotment (142) | 97753 | 53 | 1871 |
| 12 | GAD_202402081834533907 | 2024-02-08 | Memorandum | Clerk-Typist Exam 2022: main-list allotment | 114907 | 62 | 1871 |
| 13 | GAD_202405271658209807 | 2024-05-27 | Circular | Date of regular promotion to Joint Secretary | 67400 | 30 | 2301 |
| 14 | GAD_202406201431224807 | 2024-06-20 | Memorandum | Clerk-Typist Exam 2022: waiting-list allotment (155) | 92716 | 50 | 1876 |
| 15 | GAD_202412311532051007 | 2024-12-31 | Memorandum | Steno-Typist Exam 2022: main-list allotment | 92997 | 49 | 1915 |
| 16 | GAD_202504221732089007 | 2025-04-22 | GR | Review of suspended employees' cases: consolidated guidelines | 52991 | 25 | 2222 |
| 17 | GAD_202506201818570507 | 2025-06-20 | Order | Inter-departmental transfers of Clerk-Typists | 72493 | 32 | 2303 |
| 18 | GAD_202506201827169907 | 2025-06-20 | Order | Inter-departmental transfers of Asst. Section Officers | 67026 | 30 | 2284 |
| 19 | GAD_202507101245073307 | 2025-07-10 | Memorandum | Higher Grade Stenographer 2022: waiting list 1 | 73154 | 37 | 2005 |
| 20 | GAD_202507101251499107 | 2025-07-10 | Memorandum | Lower Grade Stenographer 2022: waiting list 1 | 83356 | 48 | 1762 |
| 21 | GAD_202507101257147007 | 2025-07-10 | Memorandum | Steno-Typist 2022: waiting list 1 | 90633 | 51 | 1802 |
| 22 | GAD_202512231639505907 | 2025-12-23 | GR | Navi Maharashtra Sadan: room & hall reservation/fee revision | 83602 | 39 | 2202 |
| 23 | GAD_202602021744599807 | 2026-02-02 | Memorandum | Lower Grade Stenographer 2022: waiting list 2 | 79680 | 43 | 1888 |
| 24 | GAD_202602021751282007 | 2026-02-02 | Memorandum | Steno-Typist 2022: waiting list 2 | 91342 | 47 | 1967 |
| 25 | GAD_202602161741378907 | 2026-02-16 | Memorandum | Higher Grade Stenographer 2022: waiting list 2 | 75013 | 40 | 1900 |

## Table 3. Retrieval accuracy (file-level, top-12)

| Configuration | Hit@1 | Hit@3 | Hit@12 | MRR | Mean distinct files in top-12 |
|---|---|---|---|---|---|
| Dense (bge-m3) | 64.0% (48/75) | 78.7% (59/75) | 93.3% (70/75) | 0.742 | 4.0 |
| BM25 | 77.3% (58/75) | 94.7% (71/75) | 100.0% (75/75) | 0.867 | 5.8 |
| Hybrid (RRF) | 76.0% (57/75) | 97.3% (73/75) | 100.0% (75/75) | 0.857 | 5.28 |
| Hybrid + routing | 74.7% (56/75) | 88.0% (66/75) | 88.0% (66/75) | 0.809 | 2.52 |
| Hybrid + routing + graph | 74.7% (56/75) | 88.0% (66/75) | 88.0% (66/75) | 0.809 | 2.52 |
| Hybrid + graph | 76.0% (57/75) | 97.3% (73/75) | 100.0% (75/75) | 0.857 | 5.25 |
| Hybrid, script-aware | 70.7% (53/75) | 85.3% (64/75) | 98.7% (74/75) | 0.803 | 4.69 |
| Script-aware + reranker | 84.0% (63/75) | 96.0% (72/75) | 97.3% (73/75) | 0.899 | 2.71 |

## Table 4. Retrieval by question tier

| Group | Configuration | n | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|
| detail | Dense (bge-m3) | 38 | 65.8% (25/38) | 78.9% (30/38) | 0.756 |
| detail | BM25 | 38 | 76.3% (29/38) | 94.7% (36/38) | 0.860 |
| detail | Hybrid (RRF) | 38 | 76.3% (29/38) | 97.4% (37/38) | 0.858 |
| detail | Hybrid + routing | 38 | 76.3% (29/38) | 86.8% (33/38) | 0.816 |
| detail | Hybrid + routing + graph | 38 | 76.3% (29/38) | 86.8% (33/38) | 0.816 |
| detail | Hybrid + graph | 38 | 76.3% (29/38) | 97.4% (37/38) | 0.858 |
| detail | Hybrid, script-aware | 38 | 73.7% (28/38) | 84.2% (32/38) | 0.822 |
| detail | Script-aware + reranker | 38 | 84.2% (32/38) | 97.4% (37/38) | 0.899 |
| disambiguation | Dense (bge-m3) | 20 | 40.0% (8/20) | 65.0% (13/20) | 0.571 |
| disambiguation | BM25 | 20 | 75.0% (15/20) | 95.0% (19/20) | 0.854 |
| disambiguation | Hybrid (RRF) | 20 | 60.0% (12/20) | 95.0% (19/20) | 0.765 |
| disambiguation | Hybrid + routing | 20 | 60.0% (12/20) | 85.0% (17/20) | 0.717 |
| disambiguation | Hybrid + routing + graph | 20 | 60.0% (12/20) | 85.0% (17/20) | 0.717 |
| disambiguation | Hybrid + graph | 20 | 60.0% (12/20) | 95.0% (19/20) | 0.765 |
| disambiguation | Hybrid, script-aware | 20 | 50.0% (10/20) | 75.0% (15/20) | 0.658 |
| disambiguation | Script-aware + reranker | 20 | 75.0% (15/20) | 90.0% (18/20) | 0.838 |
| situation | Dense (bge-m3) | 17 | 88.2% (15/17) | 94.1% (16/17) | 0.912 |
| situation | BM25 | 17 | 82.4% (14/17) | 94.1% (16/17) | 0.897 |
| situation | Hybrid (RRF) | 17 | 94.1% (16/17) | 100.0% (17/17) | 0.961 |
| situation | Hybrid + routing | 17 | 88.2% (15/17) | 94.1% (16/17) | 0.902 |
| situation | Hybrid + routing + graph | 17 | 88.2% (15/17) | 94.1% (16/17) | 0.902 |
| situation | Hybrid + graph | 17 | 94.1% (16/17) | 100.0% (17/17) | 0.961 |
| situation | Hybrid, script-aware | 17 | 88.2% (15/17) | 100.0% (17/17) | 0.931 |
| situation | Script-aware + reranker | 17 | 94.1% (16/17) | 100.0% (17/17) | 0.971 |

## Table 5. Retrieval by query language

| Group | Configuration | n | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|
| en | Dense (bge-m3) | 50 | 56.0% (28/50) | 72.0% (36/50) | 0.678 |
| en | BM25 | 50 | 78.0% (39/50) | 94.0% (47/50) | 0.865 |
| en | Hybrid (RRF) | 50 | 70.0% (35/50) | 96.0% (48/50) | 0.818 |
| en | Hybrid + routing | 50 | 72.0% (36/50) | 86.0% (43/50) | 0.787 |
| en | Hybrid + routing + graph | 50 | 72.0% (36/50) | 86.0% (43/50) | 0.787 |
| en | Hybrid + graph | 50 | 70.0% (35/50) | 96.0% (48/50) | 0.818 |
| en | Hybrid, script-aware | 50 | 62.0% (31/50) | 78.0% (39/50) | 0.738 |
| en | Script-aware + reranker | 50 | 80.0% (40/50) | 94.0% (47/50) | 0.868 |
| mr | Dense (bge-m3) | 25 | 80.0% (20/25) | 92.0% (23/25) | 0.870 |
| mr | BM25 | 25 | 76.0% (19/25) | 96.0% (24/25) | 0.870 |
| mr | Hybrid (RRF) | 25 | 88.0% (22/25) | 100.0% (25/25) | 0.933 |
| mr | Hybrid + routing | 25 | 80.0% (20/25) | 92.0% (23/25) | 0.853 |
| mr | Hybrid + routing + graph | 25 | 80.0% (20/25) | 92.0% (23/25) | 0.853 |
| mr | Hybrid + graph | 25 | 88.0% (22/25) | 100.0% (25/25) | 0.933 |
| mr | Hybrid, script-aware | 25 | 88.0% (22/25) | 100.0% (25/25) | 0.933 |
| mr | Script-aware + reranker | 25 | 92.0% (23/25) | 100.0% (25/25) | 0.960 |

## Table 6. End-to-end answer accuracy

| Subset | n | Fully correct | Correct or partial | Correct GR cited | Correct GR cited first | False refusal |
|---|---|---|---|---|---|---|
| All answerable | 75 | 41.3% (31/75) | 48.0% (36/75) | 100.0% (75/75) | 76.0% (57/75) | 9.3% (7/75) |
| Tier: situation | 17 | 58.8% (10/17) | 70.6% (12/17) | 100.0% (17/17) | 94.1% (16/17) | 0.0% (0/17) |
| Tier: detail | 38 | 42.1% (16/38) | 42.1% (16/38) | 100.0% (38/38) | 76.3% (29/38) | 7.9% (3/38) |
| Tier: disambiguation | 20 | 25.0% (5/20) | 40.0% (8/20) | 100.0% (20/20) | 60.0% (12/20) | 20.0% (4/20) |
| Language: English | 50 | 50.0% (25/50) | 60.0% (30/50) | 100.0% (50/50) | 70.0% (35/50) | 14.0% (7/50) |
| Language: Marathi | 25 | 24.0% (6/25) | 24.0% (6/25) | 100.0% (25/25) | 88.0% (22/25) | 0.0% (0/25) |

Negative traps (no answer in corpus): correct refusal 50.0% (3/6). Response language matched question: 97.5% (79/81). Latency median 5.32 s, p90 10.38 s, max 23.86 s.


## Table 7. Automated test suite

| Test module | Test cases | Passed | Failed/errors | Time (s) |
|---|---|---|---|---|
| test_units | 21 | 21 | 0 | 0.01 |
| test_config | 11 | 11 | 0 | 0.4 |
| test_condense | 17 | 17 | 0 | 1.48 |
| test_eval_followups | 10 | 10 | 0 | 0.13 |
| test_explained | 21 | 21 | 0 | 0.01 |
| test_files_api | 23 | 23 | 0 | 0.59 |
| test_graph | 96 | 96 | 0 | 0.05 |
| test_harness_natural | 47 | 47 | 0 | 0.51 |
| test_meta | 99 | 99 | 0 | 0.02 |
| test_quick_eval | 24 | 24 | 0 | 0.18 |
| test_reranker | 21 | 21 | 0 | 2.75 |
| test_script_aware_fusion | 14 | 14 | 0 | 0.01 |
| test_summary_v2 | 33 | 33 | 0 | 0.07 |
| TOTAL | 437 | 437 | 0 | 6.21 |

## Table 8. Functional test cases (live API)

| Test ID | Area | Test case | Input | Expected | Actual | Status | Note |
|---|---|---|---|---|---|---|---|
| FT-01 | Authentication | Login with valid credentials | admin / correct password | HTTP 200 with JWT | HTTP 200, token issued=True | Pass |  |
| FT-02 | Authentication | Login with wrong password | admin / wrong password | Rejected (401) | HTTP 401 | Pass |  |
| FT-03 | Access control | Chat query without a token | POST /api/query/chat, no Authorization | Rejected (401/403) | HTTP 401 | Pass |  |
| FT-04 | Access control | Non-admin lists users | user1 token, GET /api/auth/users | Forbidden (403) | HTTP 403 | Pass |  |
| FT-05 | Document registry | List uploaded GRs | GET /api/upload/list | 25 documents listed | HTTP 200, 25 documents | Pass |  |
| FT-06 | Evidence-first citation | Open cited source document | GET /api/files/GAD_202504221732089007.pdf.txt | HTTP 200, document served | HTTP 404, 32 bytes | Fail | Corpus is indexed as .pdf.txt text exports; api/files.py serves only *.pdf, so citation links 404 for this corpus. Add the source PDFs to data/grdocs or allow .txt. |
| FT-07 | Security | Path traversal on file endpoint | GET /api/files/..%2F.env | Rejected (4xx), no secrets | HTTP 404 | Pass |  |
| FT-08 | Robustness | Request a missing document | GET /api/files/does_not_exist.pdf | 404 Not Found | HTTP 404 | Pass |  |
| FT-09 | Health | Retrieval health check | GET /api/query/health | HTTP 200 | HTTP 200: {"success":true,"vector_ready":true,"ollama_online":true,"llm_model":"phi4-mini","embed_model":"bge-m3"} | Pass |  |
| FT-10 | Semantic search | Marathi search without LLM | POST /api/query/search 'निलंबित ... आढावा' | Suspension-review GR ranked first | HTTP 200, top=GAD_202504221732089007.pdf.txt, 1.9s | Pass |  |
| FT-11 | Summarisation | List generated summaries | GET /api/summary/list | Stored summary listed | HTTP 200, contains summary=True | Pass |  |
| FT-12 | Metadata registry | Registry metadata endpoint | GET /api/meta/ | HTTP 200 | HTTP 200, 1295 bytes | Pass |  |
| FT-13 | Bilingual answers | Answer is written in the language of the question | All 81 harness questions (en + mr) | Same script as question | 79/81 matched | Pass | 2 mismatches |
| FT-14 | Cross-lingual QA | English question about a Marathi-original GR | Who heads the Recommendation Committee that examines foreign tour proposals of state officers? | Chief Secretary; cites GAD_202102011814485107 | - The Recommendation Committee that examines foreign tour proposals of state officers is headed by the Chief Secretary. | Pass |  |
| FT-15 | Hallucination control | Out-of-corpus question is refused | What is the dearness allowance rate for state employees from January 2025? | Fixed refusal message | This information was not found in the uploaded documents. | Pass |  |
| FT-16 | Hallucination control | Abstract example: study leave (not in corpus) is refused | What is the eligibility for study leave under the Maharashtra Civil Services rules? | Fixed refusal message | - Study tours should not be conducted for general information. - Officers nearing retirement are not necessarily qualified. Moreover, it is … | Fail | Answered from the foreign-tours circular instead of refusing |
| FT-17 | Near-duplicate GRs | Pick the right memo among near-identical allotment memos | Which memorandum allotted the main-list candidates of the Clerk-Typist Examination 2022 to Mantralaya departments in February 2024? | ASC 1524/P.No.15, 8 Feb 2024 | Government Memorandum No: ASC 1524 / P.No.15 / Pr. 3 (14-A), dated February 8, 2024. | Pass |  |
| FT-18 | Near-duplicate GRs | Read a count from the second waiting-list memo | How many Lower Grade Stenographer candidates were recommended from waiting list no. 2 in February 2026? | Marathi 15, English 11 | Four | Fail | Correct GR retrieved; 3.8B model misread the count |