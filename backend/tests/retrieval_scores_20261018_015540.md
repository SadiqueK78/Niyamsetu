# Retrieval scores

- run at: 2026-10-18T01:55:40.343655+00:00
- questions: 21 from `harness_questions.json`
- top_k: 12, ROUTE_POOL_K: 30, ROUTE_TOP_DOCS: 3, ROUTE_MAX_CHUNKS: 8

Questions that do not name a document are inherently ambiguous — every GR in the corpus has a date — so these figures are a LOWER BOUND on retrieval quality. They are comparable between configurations, which is what they are for.

### Overall

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 21 |
| bm25_only | 0.429 | 0.571 | 0.667 | 0.810 | 0.546 | 8.86 | 21 |
| hybrid | 0.429 | 0.571 | 0.667 | 0.810 | 0.546 | 8.86 | 21 |
| hybrid_routed | 0.286 | 0.333 | 0.333 | 0.333 | 0.302 | 2.90 | 21 |
| hybrid_routed_graph | 0.286 | 0.333 | 0.381 | 0.381 | 0.313 | 4.05 | 21 |

### Tier: cross_reference

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 5 |
| bm25_only | 0.400 | 0.600 | 0.600 | 0.800 | 0.533 | 8.60 | 5 |
| hybrid | 0.400 | 0.600 | 0.600 | 0.800 | 0.533 | 8.60 | 5 |
| hybrid_routed | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 | 3.00 | 5 |
| hybrid_routed_graph | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 | 3.80 | 5 |

### Tier: direct_lookup

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 6 |
| bm25_only | 0.333 | 0.333 | 0.500 | 0.667 | 0.403 | 9.83 | 6 |
| hybrid | 0.333 | 0.333 | 0.500 | 0.667 | 0.403 | 9.83 | 6 |
| hybrid_routed | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 3.00 | 6 |
| hybrid_routed_graph | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 4.00 | 6 |

### Tier: negative_trap

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 5 |
| bm25_only | 0.200 | 0.600 | 0.600 | 0.800 | 0.429 | 9.40 | 5 |
| hybrid | 0.200 | 0.600 | 0.600 | 0.800 | 0.429 | 9.40 | 5 |
| hybrid_routed | 0.000 | 0.200 | 0.200 | 0.200 | 0.067 | 2.60 | 5 |
| hybrid_routed_graph | 0.000 | 0.200 | 0.400 | 0.400 | 0.117 | 3.80 | 5 |

### Tier: synthesis

| configuration | hit@1 | hit@3 | hit@5 | hit@12 | MRR | mean files | n |
|---|---|---|---|---|---|---|---|
| dense_only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 5 |
| bm25_only | 0.800 | 0.800 | 1.000 | 1.000 | 0.850 | 7.40 | 5 |
| hybrid | 0.800 | 0.800 | 1.000 | 1.000 | 0.850 | 7.40 | 5 |
| hybrid_routed | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 3.00 | 5 |
| hybrid_routed_graph | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 4.60 | 5 |

### Per question — rank of the expected file

| id | tier | expected | dense_only | bm25_only | hybrid | hybrid_routed | hybrid_routed_graph |
|---|---|---|---|---|---|---|---|
| GR1-T1 | direct_lookup | GR1.pdf | miss | 6 | 6 | miss | miss |
| GR1-T1b | direct_lookup | GR1.pdf | miss | miss | miss | miss | miss |
| GR1-T2 | synthesis | GR1.pdf | miss | 1 | 1 | 1 | 1 |
| GR1-T3 | cross_reference | GR1.pdf | miss | 1 | 1 | 1 | 1 |
| GR1-T4 | negative_trap | GR1.pdf | miss | 2 | 2 | miss | 4 |
| GR2-T1 | direct_lookup | GR2.pdf | miss | miss | miss | miss | miss |
| GR2-T2 | synthesis | GR2.pdf | miss | 4 | 4 | miss | miss |
| GR2-T3 | cross_reference | GR2.pdf | miss | miss | miss | miss | miss |
| GR2-T4 | negative_trap | GR2.pdf | miss | 7 | 7 | miss | miss |
| GR3-T1 | direct_lookup | GR3.pdf | miss | 1 | 1 | 1 | 1 |
| GR3-T2 | synthesis | GR3.pdf | miss | 1 | 1 | 1 | 1 |
| GR3-T3 | cross_reference | GR3.pdf | miss | 1 | 1 | 1 | 1 |
| GR3-T4 | negative_trap | GR3.pdf | miss | miss | miss | 3 | 3 |
| GR4-T1 | direct_lookup | GR4.pdf | miss | 4 | 4 | miss | miss |
| GR4-T2 | synthesis | GR4.pdf | miss | 1 | 1 | miss | miss |
| GR4-T3 | cross_reference | GR4.pdf | miss | 6 | 6 | miss | miss |
| GR4-T4 | negative_trap | GR4.pdf | miss | 2 | 2 | miss | miss |
| GR5-T1 | direct_lookup | GR5.pdf | miss | 1 | 1 | miss | miss |
| GR5-T2 | synthesis | GR5.pdf | miss | 1 | 1 | 1 | 1 |
| GR5-T3 | cross_reference | GR5.pdf | miss | 2 | 2 | miss | miss |
| GR5-T4 | negative_trap | GR5.pdf | miss | 1 | 1 | miss | miss |
