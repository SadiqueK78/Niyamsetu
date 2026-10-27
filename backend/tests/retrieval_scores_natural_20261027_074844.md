# Retrieval scores — natural gold set

- run at: 2026-10-27T07:48:44.075122+00:00
- questions: 75 from `gold_gad25.json` (6 negative_trap excluded)
- top_k: 12, ROUTE_POOL_K: 30, ROUTE_TOP_DOCS: 3, ROUTE_MAX_CHUNKS: 8

Each question is written to be answerable from exactly one document, so a miss here is a retrieval miss rather than an ambiguous question. page_hit@12 counts a chunk by the page it starts on. negative_trap entries (6) have no answer in the corpus and are excluded from every metric.

### Overall

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 64.0% (48/75) | 78.7% (59/75) | 93.3% (70/75) | 0.742 | 0.0% (0/75) | 4.00 | 75 |
| bm25_only | 77.3% (58/75) | 94.7% (71/75) | 100.0% (75/75) | 0.867 | 0.0% (0/75) | 5.80 | 75 |
| hybrid | 76.0% (57/75) | 97.3% (73/75) | 100.0% (75/75) | 0.857 | 0.0% (0/75) | 5.28 | 75 |
| hybrid_routed | 74.7% (56/75) | 88.0% (66/75) | 88.0% (66/75) | 0.809 | 0.0% (0/75) | 2.52 | 75 |
| hybrid_routed_graph | 74.7% (56/75) | 88.0% (66/75) | 88.0% (66/75) | 0.809 | 0.0% (0/75) | 2.52 | 75 |
| hybrid_graph | 76.0% (57/75) | 97.3% (73/75) | 100.0% (75/75) | 0.857 | 0.0% (0/75) | 5.25 | 75 |
| hybrid_script_aware | 70.7% (53/75) | 85.3% (64/75) | 98.7% (74/75) | 0.803 | 0.0% (0/75) | 4.69 | 75 |
| hybrid_script_aware_rerank | 84.0% (63/75) | 96.0% (72/75) | 97.3% (73/75) | 0.899 | 0.0% (0/75) | 2.71 | 75 |

### Tier: detail

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 65.8% (25/38) | 78.9% (30/38) | 94.7% (36/38) | 0.756 | 0.0% (0/38) | 3.63 | 38 |
| bm25_only | 76.3% (29/38) | 94.7% (36/38) | 100.0% (38/38) | 0.860 | 0.0% (0/38) | 5.92 | 38 |
| hybrid | 76.3% (29/38) | 97.4% (37/38) | 100.0% (38/38) | 0.858 | 0.0% (0/38) | 5.08 | 38 |
| hybrid_routed | 76.3% (29/38) | 86.8% (33/38) | 86.8% (33/38) | 0.816 | 0.0% (0/38) | 2.47 | 38 |
| hybrid_routed_graph | 76.3% (29/38) | 86.8% (33/38) | 86.8% (33/38) | 0.816 | 0.0% (0/38) | 2.47 | 38 |
| hybrid_graph | 76.3% (29/38) | 97.4% (37/38) | 100.0% (38/38) | 0.858 | 0.0% (0/38) | 5.08 | 38 |
| hybrid_script_aware | 73.7% (28/38) | 84.2% (32/38) | 100.0% (38/38) | 0.822 | 0.0% (0/38) | 4.47 | 38 |
| hybrid_script_aware_rerank | 84.2% (32/38) | 97.4% (37/38) | 97.4% (37/38) | 0.899 | 0.0% (0/38) | 2.37 | 38 |

### Tier: disambiguation

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 40.0% (8/20) | 65.0% (13/20) | 90.0% (18/20) | 0.571 | 0.0% (0/20) | 5.60 | 20 |
| bm25_only | 75.0% (15/20) | 95.0% (19/20) | 100.0% (20/20) | 0.854 | 0.0% (0/20) | 6.80 | 20 |
| hybrid | 60.0% (12/20) | 95.0% (19/20) | 100.0% (20/20) | 0.765 | 0.0% (0/20) | 6.80 | 20 |
| hybrid_routed | 60.0% (12/20) | 85.0% (17/20) | 85.0% (17/20) | 0.717 | 0.0% (0/20) | 2.85 | 20 |
| hybrid_routed_graph | 60.0% (12/20) | 85.0% (17/20) | 85.0% (17/20) | 0.717 | 0.0% (0/20) | 2.85 | 20 |
| hybrid_graph | 60.0% (12/20) | 95.0% (19/20) | 100.0% (20/20) | 0.765 | 0.0% (0/20) | 6.70 | 20 |
| hybrid_script_aware | 50.0% (10/20) | 75.0% (15/20) | 95.0% (19/20) | 0.658 | 0.0% (0/20) | 6.15 | 20 |
| hybrid_script_aware_rerank | 75.0% (15/20) | 90.0% (18/20) | 95.0% (19/20) | 0.838 | 0.0% (0/20) | 3.70 | 20 |

### Tier: situation

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 88.2% (15/17) | 94.1% (16/17) | 94.1% (16/17) | 0.912 | 0.0% (0/17) | 2.94 | 17 |
| bm25_only | 82.4% (14/17) | 94.1% (16/17) | 100.0% (17/17) | 0.897 | 0.0% (0/17) | 4.35 | 17 |
| hybrid | 94.1% (16/17) | 100.0% (17/17) | 100.0% (17/17) | 0.961 | 0.0% (0/17) | 3.94 | 17 |
| hybrid_routed | 88.2% (15/17) | 94.1% (16/17) | 94.1% (16/17) | 0.902 | 0.0% (0/17) | 2.24 | 17 |
| hybrid_routed_graph | 88.2% (15/17) | 94.1% (16/17) | 94.1% (16/17) | 0.902 | 0.0% (0/17) | 2.24 | 17 |
| hybrid_graph | 94.1% (16/17) | 100.0% (17/17) | 100.0% (17/17) | 0.961 | 0.0% (0/17) | 3.94 | 17 |
| hybrid_script_aware | 88.2% (15/17) | 100.0% (17/17) | 100.0% (17/17) | 0.931 | 0.0% (0/17) | 3.47 | 17 |
| hybrid_script_aware_rerank | 94.1% (16/17) | 100.0% (17/17) | 100.0% (17/17) | 0.971 | 0.0% (0/17) | 2.29 | 17 |

## By question language

### Language: en

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 56.0% (28/50) | 72.0% (36/50) | 92.0% (46/50) | 0.678 | 0.0% (0/50) | 4.16 | 50 |
| bm25_only | 78.0% (39/50) | 94.0% (47/50) | 100.0% (50/50) | 0.865 | 0.0% (0/50) | 5.82 | 50 |
| hybrid | 70.0% (35/50) | 96.0% (48/50) | 100.0% (50/50) | 0.818 | 0.0% (0/50) | 5.28 | 50 |
| hybrid_routed | 72.0% (36/50) | 86.0% (43/50) | 86.0% (43/50) | 0.787 | 0.0% (0/50) | 2.48 | 50 |
| hybrid_routed_graph | 72.0% (36/50) | 86.0% (43/50) | 86.0% (43/50) | 0.787 | 0.0% (0/50) | 2.48 | 50 |
| hybrid_graph | 70.0% (35/50) | 96.0% (48/50) | 100.0% (50/50) | 0.818 | 0.0% (0/50) | 5.24 | 50 |
| hybrid_script_aware | 62.0% (31/50) | 78.0% (39/50) | 98.0% (49/50) | 0.738 | 0.0% (0/50) | 4.40 | 50 |
| hybrid_script_aware_rerank | 80.0% (40/50) | 94.0% (47/50) | 96.0% (48/50) | 0.868 | 0.0% (0/50) | 2.78 | 50 |

### Language: mr

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 80.0% (20/25) | 92.0% (23/25) | 96.0% (24/25) | 0.870 | 0.0% (0/25) | 3.68 | 25 |
| bm25_only | 76.0% (19/25) | 96.0% (24/25) | 100.0% (25/25) | 0.870 | 0.0% (0/25) | 5.76 | 25 |
| hybrid | 88.0% (22/25) | 100.0% (25/25) | 100.0% (25/25) | 0.933 | 0.0% (0/25) | 5.28 | 25 |
| hybrid_routed | 80.0% (20/25) | 92.0% (23/25) | 92.0% (23/25) | 0.853 | 0.0% (0/25) | 2.60 | 25 |
| hybrid_routed_graph | 80.0% (20/25) | 92.0% (23/25) | 92.0% (23/25) | 0.853 | 0.0% (0/25) | 2.60 | 25 |
| hybrid_graph | 88.0% (22/25) | 100.0% (25/25) | 100.0% (25/25) | 0.933 | 0.0% (0/25) | 5.28 | 25 |
| hybrid_script_aware | 88.0% (22/25) | 100.0% (25/25) | 100.0% (25/25) | 0.933 | 0.0% (0/25) | 5.28 | 25 |
| hybrid_script_aware_rerank | 92.0% (23/25) | 100.0% (25/25) | 100.0% (25/25) | 0.960 | 0.0% (0/25) | 2.56 | 25 |

## Reranked (hybrid_script_aware_rerank)

BAAI/bge-reranker-v2-m3: 20 candidates -> top 5. hit@12 is over the 5-chunk list that exists. Reranker fallbacks to the original order: 0 of 75 calls.

### Overall

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 84.0% (63/75) | 96.0% (72/75) | 97.3% (73/75) | 97.3% (73/75) | 0.899 | 0.0% (0/75) | 2.71 | 75 |

### Tier: detail

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 84.2% (32/38) | 97.4% (37/38) | 97.4% (37/38) | 97.4% (37/38) | 0.899 | 0.0% (0/38) | 2.37 | 38 |

### Tier: disambiguation

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 75.0% (15/20) | 90.0% (18/20) | 95.0% (19/20) | 95.0% (19/20) | 0.838 | 0.0% (0/20) | 3.70 | 20 |

### Tier: situation

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 94.1% (16/17) | 100.0% (17/17) | 100.0% (17/17) | 100.0% (17/17) | 0.971 | 0.0% (0/17) | 2.29 | 17 |

### Language: en

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 80.0% (40/50) | 94.0% (47/50) | 96.0% (48/50) | 96.0% (48/50) | 0.868 | 0.0% (0/50) | 2.78 | 50 |

### Language: mr

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 | mean files | n |
|---|---|---|---|---|---|---|---|---|
| hybrid_script_aware_rerank | 92.0% (23/25) | 100.0% (25/25) | 100.0% (25/25) | 100.0% (25/25) | 0.960 | 0.0% (0/25) | 2.56 | 25 |

### Per question — rank of the expected file

`p` after a rank: a chunk from the expected page was among the first 12 retrieved chunks.

| id | tier | expected | dense_only | bm25_only | hybrid | hybrid_routed | hybrid_routed_graph | hybrid_graph | hybrid_script_aware | hybrid_script_aware_rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| gad-001 | situation | GAD_201811161110169207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-002 | detail | GAD_201811161110169207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-003 | detail | GAD_201811161110169207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-004 | situation | GAD_201812201818380207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-005 | detail | GAD_201812201818380207.pdf.txt | miss | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-006 | detail | GAD_201812201818380207.pdf.txt | miss | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-007 | situation | GAD_202102011814485107.pdf.txt | miss | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-008 | detail | GAD_202102011814485107.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-009 | detail | GAD_202102011814485107.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-010 | situation | GAD_202102161256152607.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-011 | detail | GAD_202102161256152607.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-012 | situation | GAD_202102161256152607.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-013 | situation | GAD_202105201508210107.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-014 | detail | GAD_202105201508210107.pdf.txt | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| gad-015 | detail | GAD_202105201508210107.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-016 | situation | GAD_202108301543421007.pdf.txt | 1 | 1 | 1 | miss | miss | 1 | 1 | 1 |
| gad-017 | detail | GAD_202108301543421007.pdf.txt | 1 | 1 | 1 | miss | miss | 1 | 1 | 1 |
| gad-018 | disambiguation | GAD_202108301543421007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-019 | situation | GAD_202109011815562507.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-020 | detail | GAD_202109011815562507.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-021 | disambiguation | GAD_202109011815562507.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-022 | situation | GAD_202205041847101207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-023 | detail | GAD_202205041847101207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-024 | detail | GAD_202205041847101207.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-025 | disambiguation | GAD_202305111444322707.pdf.txt | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| gad-026 | detail | GAD_202305111444322707.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-027 | detail | GAD_202305111444322707.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-028 | disambiguation | GAD_202305111455014207.pdf.txt | 4 | 2 | 3 | 2 | 2 | 3 | 4 | 1 |
| gad-029 | detail | GAD_202305111455014207.pdf.txt | 5 | 1 | 2 | 1 | 1 | 2 | 5 | 1 |
| gad-030 | detail | GAD_202305111455014207.pdf.txt | 2 | 2 | 1 | miss | miss | 1 | 1 | 1 |
| gad-031 | disambiguation | GAD_202401091710039807.pdf.txt | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| gad-032 | detail | GAD_202401091710039807.pdf.txt | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-033 | situation | GAD_202401091710039807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |
| gad-034 | disambiguation | GAD_202402081834533907.pdf.txt | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 1 |
| gad-035 | detail | GAD_202402081834533907.pdf.txt | 1 | 3 | 2 | 2 | 2 | 2 | 1 | 2 |
| gad-036 | detail | GAD_202402081834533907.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |
| gad-037 | situation | GAD_202405271658209807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-038 | detail | GAD_202405271658209807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-039 | detail | GAD_202405271658209807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-040 | disambiguation | GAD_202406201431224807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-041 | detail | GAD_202406201431224807.pdf.txt | 4 | 1 | 2 | 2 | 2 | 2 | 4 | 1 |
| gad-042 | disambiguation | GAD_202406201431224807.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-043 | disambiguation | GAD_202412311532051007.pdf.txt | miss | 3 | 7 | miss | miss | 7 | miss | miss |
| gad-044 | detail | GAD_202412311532051007.pdf.txt | 8 | 7 | 9 | miss | miss | 9 | 8 | 3 |
| gad-045 | situation | GAD_202412311532051007.pdf.txt | 1 | 4 | 3 | 3 | 3 | 3 | 3 | 1 |
| gad-046 | situation | GAD_202504221732089007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-047 | detail | GAD_202504221732089007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-048 | detail | GAD_202504221732089007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-049 | disambiguation | GAD_202506201818570507.pdf.txt | miss | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-050 | detail | GAD_202506201818570507.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-051 | detail | GAD_202506201818570507.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-052 | disambiguation | GAD_202506201827169907.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-053 | detail | GAD_202506201827169907.pdf.txt | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| gad-054 | detail | GAD_202506201827169907.pdf.txt | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-055 | disambiguation | GAD_202507101245073307.pdf.txt | 4 | 1 | 2 | 1 | 1 | 2 | 4 | 1 |
| gad-056 | situation | GAD_202507101245073307.pdf.txt | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-057 | detail | GAD_202507101245073307.pdf.txt | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 |
| gad-058 | disambiguation | GAD_202507101251499107.pdf.txt | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 2 |
| gad-059 | situation | GAD_202507101251499107.pdf.txt | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-060 | detail | GAD_202507101251499107.pdf.txt | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 3 |
| gad-061 | disambiguation | GAD_202507101257147007.pdf.txt | 4 | 1 | 1 | miss | miss | 1 | 4 | 4 |
| gad-062 | situation | GAD_202507101257147007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-063 | detail | GAD_202507101257147007.pdf.txt | 6 | 2 | 3 | miss | miss | 3 | 6 | miss |
| gad-064 | situation | GAD_202512231639505907.pdf.txt | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| gad-065 | detail | GAD_202512231639505907.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-066 | detail | GAD_202512231639505907.pdf.txt | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| gad-067 | disambiguation | GAD_202602021744599807.pdf.txt | 3 | 1 | 2 | 2 | 2 | 2 | 3 | 2 |
| gad-068 | disambiguation | GAD_202602021744599807.pdf.txt | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 |
| gad-069 | detail | GAD_202602021744599807.pdf.txt | 4 | 1 | 3 | 2 | 2 | 3 | 4 | 1 |
| gad-070 | disambiguation | GAD_202602021751282007.pdf.txt | 1 | 1 | 1 | miss | miss | 1 | 1 | 1 |
| gad-071 | disambiguation | GAD_202602021751282007.pdf.txt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-072 | detail | GAD_202602021751282007.pdf.txt | 2 | 1 | 1 | miss | miss | 1 | 2 | 1 |
| gad-073 | disambiguation | GAD_202602161741378907.pdf.txt | 4 | 1 | 2 | 1 | 1 | 2 | 4 | 2 |
| gad-074 | disambiguation | GAD_202602161741378907.pdf.txt | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gad-075 | detail | GAD_202602161741378907.pdf.txt | 4 | 1 | 3 | 1 | 1 | 3 | 4 | 1 |
