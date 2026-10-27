# Retrieval scores

- run at: 2026-10-18T02:47:00.369257+00:00
- questions: 21 from `harness_questions.json`
- top_k: 12, ROUTE_POOL_K: 30, ROUTE_TOP_DOCS: 3, ROUTE_MAX_CHUNKS: 8

Questions that do not name a document are inherently ambiguous — every GR in the corpus has a date — so these figures are a LOWER BOUND on retrieval quality. They are comparable between configurations, which is what they are for.

### Overall

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.571 | 0.714 | 0.714 | 0.810 | 0.650 | 6.52 | 21 |
| bm25_only | 0.429 | 0.571 | 0.667 | 0.810 | 0.546 | 8.86 | 21 |
| hybrid | 0.714 | 0.762 | 0.762 | 0.857 | 0.751 | 8.14 | 21 |
| hybrid_routed | 0.619 | 0.619 | 0.619 | 0.619 | 0.619 | 2.81 | 21 |
| hybrid_routed_graph | 0.619 | 0.619 | 0.619 | 0.619 | 0.619 | 3.24 | 21 |
| hybrid_graph | 0.714 | 0.762 | 0.762 | 0.857 | 0.751 | 8.19 | 21 |

### Tier: cross_reference

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.400 | 0.600 | 0.600 | 0.800 | 0.529 | 7.00 | 5 |
| bm25_only | 0.400 | 0.600 | 0.600 | 0.800 | 0.533 | 8.60 | 5 |
| hybrid | 0.600 | 0.600 | 0.600 | 1.000 | 0.656 | 7.60 | 5 |
| hybrid_routed | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 | 2.60 | 5 |
| hybrid_routed_graph | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 | 3.20 | 5 |
| hybrid_graph | 0.600 | 0.600 | 0.600 | 1.000 | 0.656 | 7.60 | 5 |

### Tier: direct_lookup

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.167 | 0.333 | 0.333 | 0.500 | 0.278 | 7.50 | 6 |
| bm25_only | 0.333 | 0.333 | 0.500 | 0.667 | 0.403 | 9.83 | 6 |
| hybrid | 0.333 | 0.500 | 0.500 | 0.500 | 0.417 | 9.33 | 6 |
| hybrid_routed | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 3.00 | 6 |
| hybrid_routed_graph | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 3.33 | 6 |
| hybrid_graph | 0.333 | 0.500 | 0.500 | 0.500 | 0.417 | 9.50 | 6 |

### Tier: negative_trap

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.800 | 1.000 | 1.000 | 1.000 | 0.867 | 5.60 | 5 |
| bm25_only | 0.200 | 0.600 | 0.600 | 0.800 | 0.429 | 9.40 | 5 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.40 | 5 |
| hybrid_routed | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.00 | 5 |
| hybrid_routed_graph | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.60 | 5 |
| hybrid_graph | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.40 | 5 |

### Tier: synthesis

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.80 | 5 |
| bm25_only | 0.800 | 0.800 | 1.000 | 1.000 | 0.850 | 7.40 | 5 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.00 | 5 |
| hybrid_routed | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.60 | 5 |
| hybrid_routed_graph | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.80 | 5 |
| hybrid_graph | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.00 | 5 |

### Per question — rank of the expected file

| id | tier | expected | dense_only | bm25_only | hybrid | hybrid_routed | hybrid_routed_graph | hybrid_graph |
|---|---|---|---|---|---|---|---|---|
| GR1-T1 | direct_lookup | GR1.pdf | miss | 6 | miss | miss | miss | miss |
| GR1-T1b | direct_lookup | GR1.pdf | miss | miss | miss | miss | miss | miss |
| GR1-T2 | synthesis | GR1.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR1-T3 | cross_reference | GR1.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR1-T4 | negative_trap | GR1.pdf | 3 | 2 | 1 | 1 | 1 | 1 |
| GR2-T1 | direct_lookup | GR2.pdf | miss | miss | miss | miss | miss | miss |
| GR2-T2 | synthesis | GR2.pdf | 1 | 4 | 1 | 1 | 1 | 1 |
| GR2-T3 | cross_reference | GR2.pdf | miss | miss | 9 | miss | miss | 9 |
| GR2-T4 | negative_trap | GR2.pdf | 1 | 7 | 1 | 1 | 1 | 1 |
| GR3-T1 | direct_lookup | GR3.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR3-T2 | synthesis | GR3.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR3-T3 | cross_reference | GR3.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR3-T4 | negative_trap | GR3.pdf | 1 | miss | 1 | 1 | 1 | 1 |
| GR4-T1 | direct_lookup | GR4.pdf | 2 | 4 | 2 | miss | miss | 2 |
| GR4-T2 | synthesis | GR4.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR4-T3 | cross_reference | GR4.pdf | 7 | 6 | 6 | miss | miss | 6 |
| GR4-T4 | negative_trap | GR4.pdf | 1 | 2 | 1 | 1 | 1 | 1 |
| GR5-T1 | direct_lookup | GR5.pdf | 6 | 1 | 1 | miss | miss | 1 |
| GR5-T2 | synthesis | GR5.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
| GR5-T3 | cross_reference | GR5.pdf | 2 | 2 | 1 | miss | miss | 1 |
| GR5-T4 | negative_trap | GR5.pdf | 1 | 1 | 1 | 1 | 1 | 1 |
