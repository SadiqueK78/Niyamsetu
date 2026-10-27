# Retrieval scores — natural gold set

- run at: 2026-10-25T15:58:12.481499+00:00
- questions: 57 from `gold_natural.json` (6 negative_trap excluded)
- top_k: 12, ROUTE_POOL_K: 30, ROUTE_TOP_DOCS: 3, ROUTE_MAX_CHUNKS: 8

Each question is written to be answerable from exactly one document, so a miss here is a retrieval miss rather than an ambiguous question. page_hit@12 counts a chunk by the page it starts on. negative_trap entries (6) have no answer in the corpus and are excluded from every metric.

### Overall

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 84.2% (48/57) | 98.2% (56/57) | 98.2% (56/57) | 0.909 | 98.2% (56/57) | 4.93 | 57 |
| bm25_only | 70.2% (40/57) | 75.4% (43/57) | 77.2% (44/57) | 0.731 | 73.7% (42/57) | 5.58 | 57 |
| hybrid | 77.2% (44/57) | 82.5% (47/57) | 82.5% (47/57) | 0.789 | 78.9% (45/57) | 4.93 | 57 |
| hybrid_routed | 61.4% (35/57) | 61.4% (35/57) | 61.4% (35/57) | 0.614 | 57.9% (33/57) | 2.40 | 57 |
| hybrid_routed_graph | 61.4% (35/57) | 64.9% (37/57) | 66.7% (38/57) | 0.630 | 63.2% (36/57) | 2.79 | 57 |
| hybrid_graph | 77.2% (44/57) | 82.5% (47/57) | 82.5% (47/57) | 0.789 | 78.9% (45/57) | 5.12 | 57 |
| hybrid_script_aware | 93.0% (53/57) | 98.2% (56/57) | 98.2% (56/57) | 0.956 | 98.2% (56/57) | 5.47 | 57 |

### Tier: detail

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 87.5% (21/24) | 100.0% (24/24) | 100.0% (24/24) | 0.931 | 100.0% (24/24) | 4.54 | 24 |
| bm25_only | 54.2% (13/24) | 58.3% (14/24) | 58.3% (14/24) | 0.562 | 50.0% (12/24) | 4.25 | 24 |
| hybrid | 58.3% (14/24) | 66.7% (16/24) | 66.7% (16/24) | 0.611 | 58.3% (14/24) | 3.83 | 24 |
| hybrid_routed | 50.0% (12/24) | 50.0% (12/24) | 50.0% (12/24) | 0.500 | 41.7% (10/24) | 2.25 | 24 |
| hybrid_routed_graph | 50.0% (12/24) | 50.0% (12/24) | 54.2% (13/24) | 0.510 | 45.8% (11/24) | 2.38 | 24 |
| hybrid_graph | 58.3% (14/24) | 66.7% (16/24) | 66.7% (16/24) | 0.611 | 58.3% (14/24) | 3.83 | 24 |
| hybrid_script_aware | 91.7% (22/24) | 100.0% (24/24) | 100.0% (24/24) | 0.958 | 100.0% (24/24) | 4.83 | 24 |

### Tier: relationship

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 71.4% (10/14) | 92.9% (13/14) | 92.9% (13/14) | 0.821 | 92.9% (13/14) | 5.43 | 14 |
| bm25_only | 78.6% (11/14) | 78.6% (11/14) | 85.7% (12/14) | 0.798 | 85.7% (12/14) | 6.14 | 14 |
| hybrid | 85.7% (12/14) | 85.7% (12/14) | 85.7% (12/14) | 0.857 | 85.7% (12/14) | 5.43 | 14 |
| hybrid_routed | 57.1% (8/14) | 57.1% (8/14) | 57.1% (8/14) | 0.571 | 57.1% (8/14) | 2.50 | 14 |
| hybrid_routed_graph | 57.1% (8/14) | 64.3% (9/14) | 64.3% (9/14) | 0.595 | 64.3% (9/14) | 3.29 | 14 |
| hybrid_graph | 85.7% (12/14) | 85.7% (12/14) | 85.7% (12/14) | 0.857 | 85.7% (12/14) | 5.86 | 14 |
| hybrid_script_aware | 85.7% (12/14) | 92.9% (13/14) | 92.9% (13/14) | 0.893 | 92.9% (13/14) | 5.57 | 14 |

### Tier: situation

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 89.5% (17/19) | 100.0% (19/19) | 100.0% (19/19) | 0.947 | 100.0% (19/19) | 5.05 | 19 |
| bm25_only | 84.2% (16/19) | 94.7% (18/19) | 94.7% (18/19) | 0.895 | 94.7% (18/19) | 6.84 | 19 |
| hybrid | 94.7% (18/19) | 100.0% (19/19) | 100.0% (19/19) | 0.965 | 100.0% (19/19) | 5.95 | 19 |
| hybrid_routed | 78.9% (15/19) | 78.9% (15/19) | 78.9% (15/19) | 0.789 | 78.9% (15/19) | 2.53 | 19 |
| hybrid_routed_graph | 78.9% (15/19) | 84.2% (16/19) | 84.2% (16/19) | 0.807 | 84.2% (16/19) | 2.95 | 19 |
| hybrid_graph | 94.7% (18/19) | 100.0% (19/19) | 100.0% (19/19) | 0.965 | 100.0% (19/19) | 6.21 | 19 |
| hybrid_script_aware | 100.0% (19/19) | 100.0% (19/19) | 100.0% (19/19) | 1.000 | 100.0% (19/19) | 6.21 | 19 |

## By question language

### Language: en

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 81.0% (17/21) | 95.2% (20/21) | 95.2% (20/21) | 0.881 | 95.2% (20/21) | 3.90 | 21 |
| bm25_only | 33.3% (7/21) | 38.1% (8/21) | 38.1% (8/21) | 0.357 | 28.6% (6/21) | 2.19 | 21 |
| hybrid | 38.1% (8/21) | 52.4% (11/21) | 52.4% (11/21) | 0.429 | 42.9% (9/21) | 2.43 | 21 |
| hybrid_routed | 38.1% (8/21) | 38.1% (8/21) | 38.1% (8/21) | 0.381 | 28.6% (6/21) | 1.90 | 21 |
| hybrid_routed_graph | 38.1% (8/21) | 38.1% (8/21) | 38.1% (8/21) | 0.381 | 28.6% (6/21) | 1.90 | 21 |
| hybrid_graph | 38.1% (8/21) | 52.4% (11/21) | 52.4% (11/21) | 0.429 | 42.9% (9/21) | 2.43 | 21 |
| hybrid_script_aware | 81.0% (17/21) | 95.2% (20/21) | 95.2% (20/21) | 0.881 | 95.2% (20/21) | 3.90 | 21 |

### Language: mr

| configuration | hit@1 | hit@3 | hit@12 | MRR | page_hit@12 | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 86.1% (31/36) | 100.0% (36/36) | 100.0% (36/36) | 0.926 | 100.0% (36/36) | 5.53 | 36 |
| bm25_only | 91.7% (33/36) | 97.2% (35/36) | 100.0% (36/36) | 0.949 | 100.0% (36/36) | 7.56 | 36 |
| hybrid | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | 1.000 | 100.0% (36/36) | 6.39 | 36 |
| hybrid_routed | 75.0% (27/36) | 75.0% (27/36) | 75.0% (27/36) | 0.750 | 75.0% (27/36) | 2.69 | 36 |
| hybrid_routed_graph | 75.0% (27/36) | 80.6% (29/36) | 83.3% (30/36) | 0.775 | 83.3% (30/36) | 3.31 | 36 |
| hybrid_graph | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | 1.000 | 100.0% (36/36) | 6.69 | 36 |
| hybrid_script_aware | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | 1.000 | 100.0% (36/36) | 6.39 | 36 |

### Per question — rank of the expected file

`p` after a rank: a chunk from the expected page was among the first 12 retrieved chunks.

| id | tier | expected | dense_only | bm25_only | hybrid | hybrid_routed | hybrid_routed_graph | hybrid_graph | hybrid_script_aware |
|---|---|---|---|---|---|---|---|---|---|
| nat-001 | situation | GR1.1.pdf p7 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-002 | detail | GR1.1.pdf p9 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-003 | detail | GR1.1.pdf p6 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-004 | situation | GR1.2.pdf p4 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-005 | detail | GR1.2.pdf p3 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-006 | detail | GR1.2.pdf p4 | 1 p | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-007 | situation | GR1.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-008 | detail | GR1.pdf p2 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-009 | relationship | GR1.pdf p1 | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-010 | situation | GR1.4.pdf p2 | 1 p | 1 p | 1 p | miss | 3 p | 1 p | 1 p |
| nat-011 | detail | GR1.4.pdf p1 | 2 p | miss | miss | miss | miss | miss | 2 p |
| nat-012 | relationship | GR1.4.pdf p1 | 1 p | 1 p | 1 p | miss | 3 p | 1 p | 1 p |
| nat-013 | situation | GR1.5.pdf p2 | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-014 | detail | GR1.5.pdf p2 | 3 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-015 | relationship | GR1.5.pdf p1 | miss | miss | miss | miss | miss | miss | miss |
| nat-016 | situation | GR2.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-017 | detail | GR2.pdf p1 | 1 p | 1 | 1 | 1 | 1 | 1 | 1 p |
| nat-018 | detail | GR2.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-019 | situation | GR3.pdf p2 | 1 p | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-020 | detail | GR3.pdf p3 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-021 | relationship | GR3.pdf p1 | 1 p | 6 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-022 | situation | GR4.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-023 | detail | GR4.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-024 | relationship | GR4.pdf p1 | 2 p | miss | miss | miss | miss | miss | 2 p |
| nat-025 | situation | GR5.pdf p1 | 1 p | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-026 | detail | GR5.pdf p1 | 1 p | miss | 3 p | miss | miss | 3 p | 1 p |
| nat-027 | relationship | GR5.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-028 | situation | aadesh 1.pdf p1 | 2 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-029 | detail | aadesh 1.pdf p1 | 1 p | miss | 3 p | miss | miss | 3 p | 1 p |
| nat-030 | relationship | aadesh 1.pdf p1 | 1 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-031 | situation | aadesh 2.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-032 | detail | aadesh 2.pdf p1 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-033 | detail | aadesh 2.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-034 | situation | aadesh 3.pdf p4 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-035 | detail | aadesh 3.pdf p1 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-036 | detail | aadesh 3.pdf p4 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-037 | situation | aadesh 4.pdf p1 | 1 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-038 | detail | aadesh 4.pdf p1 | 1 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-039 | relationship | aadesh 4.pdf p1 | 1 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-040 | situation | aadesh 5.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-041 | detail | aadesh 5.pdf p1 | 2 p | miss | miss | miss | miss | miss | 2 p |
| nat-042 | relationship | aadesh 5.pdf p1 | 2 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-043 | situation | letter 1.pdf p2 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-044 | detail | letter 1.pdf p2 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-045 | relationship | letter 1.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-046 | situation | letter 2.pdf p1 | 1 p | miss | 3 p | miss | miss | 3 p | 1 p |
| nat-047 | detail | letter 2.pdf p1 | 1 p | 1 p | 1 p | miss | 4 p | 1 p | 1 p |
| nat-048 | relationship | letter 2.pdf p1 | 1 p | 1 p | 1 p | miss | miss | 1 p | 1 p |
| nat-049 | situation | letter 3.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-050 | detail | letter 3.pdf p1 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-051 | relationship | letter 3.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-052 | situation | paripatrika 1.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-053 | detail | paripatrika 1.pdf p1 | 1 p | miss | miss | miss | miss | miss | 1 p |
| nat-054 | relationship | paripatrika 1.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-055 | situation | paripatrika 2.pdf p5 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
| nat-056 | detail | paripatrika 2.pdf p4 | 1 p | 1 | 1 | 1 | 1 | 1 | 1 p |
| nat-057 | relationship | paripatrika 2.pdf p1 | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p |
