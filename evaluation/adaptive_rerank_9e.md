# Phase 9E adaptive-rerank diagnostic

**Status: VALID** - descriptive diagnostic over the frozen Phase 8 set; not independent validation; no automatic winner; production config unchanged.

## Counterfactual classes (answerable)
{'RETRIEVAL_MISS': 5, 'HELPED': 17, 'HURT': 8, 'NEUTRAL': 14}; signs_conflict: ['q41']

## Per project
- P130544: {'questions': 24, 'RETRIEVAL_MISS': 1, 'HELPED': 8, 'HURT': 5, 'NEUTRAL': 10, 'candidate_generation_miss_rate': 0.0417, 'ranking_miss_rate_p0_50': 0.0833, 'ranking_miss_rate_p1': 0.0417}
- P179039: {'questions': 11, 'RETRIEVAL_MISS': 4, 'HELPED': 4, 'HURT': 0, 'NEUTRAL': 3, 'candidate_generation_miss_rate': 0.3636, 'ranking_miss_rate_p0_50': 0.0909, 'ranking_miss_rate_p1': 0.0909}
- P506272: {'questions': 9, 'RETRIEVAL_MISS': 0, 'HELPED': 5, 'HURT': 3, 'NEUTRAL': 1, 'candidate_generation_miss_rate': 0.0, 'ranking_miss_rate_p0_50': 0.1111, 'ranking_miss_rate_p1': 0.0}

## Frontier (every preregistered point)
| point | R@5 | R@10 | MRR | nDCG@5 | rate | help capture | hurt exp. | ret. MRR | ret. nDCG | dMRR vs rnd | dnDCG vs rnd | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 (k10, historical) | 0.7727 | 0.7955 | 0.5904 | 0.6415 | - | - | - | - | - | - | - | 115.8 | 153.2 |
| P0@50 | 0.7727 | 0.7955 | 0.597 | 0.6499 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 121.1 | 178.7 |
| P1 | 0.7955 | 0.8409 | 0.6827 | 0.7169 | 1.0 | 1.0 | 8 | 1.0 | 1.0 | 0.0 | 0.0 | 5228.3 | 6126.6 |
| P2 | 0.8182 | 0.8636 | 0.7105 | 0.7366 | 0.7347 | 0.9412 | 6 | 1.3244 | 1.294 | 0.0531 | 0.0395 | 5045.7 | 6126.7 |
| P3(0.2) | 0.8182 | 0.8409 | 0.6152 | 0.6781 | 0.2449 | 0.2353 | 3 | 0.2124 | 0.4209 | -0.0013 | 0.013 | 130.0 | 6091.5 |
| P3(0.3) | 0.8182 | 0.8409 | 0.6164 | 0.6781 | 0.3061 | 0.2941 | 3 | 0.2264 | 0.4209 | -0.0059 | 0.0084 | 132.1 | 6126.7 |
| P3(0.4) | 0.8182 | 0.8409 | 0.6372 | 0.6941 | 0.4898 | 0.5294 | 4 | 0.4691 | 0.6597 | 0.0012 | 0.0137 | 178.9 | 6126.7 |
| P3(0.5) | 0.8182 | 0.8409 | 0.6561 | 0.7031 | 0.6327 | 0.7059 | 5 | 0.6896 | 0.794 | 0.0084 | 0.0136 | 4977.3 | 6126.7 |
| P4(3) | 0.8182 | 0.8409 | 0.6565 | 0.7006 | 0.449 | 0.5294 | 4 | 0.6943 | 0.7567 | 0.0205 | 0.0202 | 159.2 | 6050.5 |
| P4(4) | 0.7955 | 0.8182 | 0.6421 | 0.6844 | 0.2041 | 0.2941 | 3 | 0.5263 | 0.5149 | 0.0276 | 0.0208 | 130.0 | 5756.7 |
| P4(5) | 0.7727 | 0.7955 | 0.605 | 0.6471 | 0.1224 | 0.1176 | 2 | 0.0933 | -0.0418 | -0.0017 | -0.0104 | 122.0 | 5740.4 |
| P5(0.3) | 0.7955 | 0.8182 | 0.6362 | 0.6856 | 0.0612 | 0.1176 | 1 | 0.4574 | 0.5328 | 0.0334 | 0.0311 | 122.0 | 2915.6 |
| P5(0.5) | 0.7727 | 0.8182 | 0.6497 | 0.6911 | 0.2041 | 0.2941 | 2 | 0.6149 | 0.6149 | 0.0352 | 0.0275 | 130.0 | 5401.7 |
| P5(0.7) | 0.7955 | 0.8409 | 0.6994 | 0.7302 | 0.5918 | 0.7647 | 3 | 1.1949 | 1.1985 | 0.0517 | 0.0407 | 2014.5 | 5973.8 |
| P6 | 0.7955 | 0.8182 | 0.6367 | 0.6727 | 0.3673 | 0.3529 | 3 | 0.4632 | 0.3403 | 0.0085 | -0.0016 | 146.0 | 5950.5 |

## Oracle - USES GROUND TRUTH — NOT DEPLOYABLE
rate 0.3469, MRR 0.7525, nDCG@5 0.7898, retained MRR 1.8145, retained nDCG 2.0881

## Pareto sets (deployable points)
- mrr_vs_rerank_rate: ['P0@50', 'P2', 'P4(3)', 'P5(0.3)', 'P5(0.5)', 'P5(0.7)']
- ndcg_at_5_vs_rerank_rate: ['P0@50', 'P2', 'P4(3)', 'P5(0.3)', 'P5(0.5)', 'P5(0.7)']
- mrr_vs_composed_p95_ms: ['P0@50', 'P2', 'P5(0.3)', 'P5(0.5)', 'P5(0.7)']

## Live latency validation point (rerank rate rule; NOT a production winner)
{'point': 'P3(0.4)', 'policy_id': 'P3', 'threshold': 0.4, 'rerank_rate': 0.4898, 'rule': 'closest rerank rate to 0.5; ties: lower policy id, then lower threshold', 'label': 'latency validation point - NOT a production winner'}
