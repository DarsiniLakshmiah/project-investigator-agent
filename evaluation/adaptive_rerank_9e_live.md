# Phase 9E live latency confirmation

Selection: {'point': 'P3(0.4)', 'policy_id': 'P3', 'threshold': 0.4, 'rerank_rate': 0.4898, 'rule': 'closest rerank rate to 0.5; ties: lower policy id, then lower threshold', 'label': 'latency validation point - NOT a production winner', 'lock_sha256': '480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e', 'collection_artifact_sha256': '8b6763e349215899b686fff45f49eb44483f65490167b6649e88014599a63d70'} - latency validation point - NOT a production winner

| point | composed p50 | composed p95 | live p50 | live p95 | d p50 | d p95 | mism. |
|---|---|---|---|---|---|---|---|
| P0@50 | 121.1 | 178.7 | 120.9 | 169.4 | -0.2 | -9.3 | 0 |
| P3(0.4) | 178.9 | 6126.7 | 196.9 | 6378.3 | 18.0 | 251.6 | 0 |

## P0@50

- relative difference: p50 -0.0017, p95 -0.052
- observed rerank rate (timed passes): 0.0
- agreement over 147 rows: 0 decision, 0 top-5 mismatches
- live reranked: n=0 p50 None p95 None
- live not_reranked: n=49 p50 120.9 p95 169.4
- pass 1 (warm-up): p50 1596.4 p95 1975.7
- pass 2 (timed): p50 113.8 p95 185.4
- pass 3 (timed): p50 114.3 p95 170.1

| question | rerank | composed ms | live ms | diff ms |
|---|---|---|---|---|
| q01 | False | 124.4 | 169.4 | 45.1 |
| q02 | False | 121.9 | 128.8 | 6.9 |
| q03 | False | 228.1 | 120.5 | -107.6 |
| q04 | False | 104.9 | 169.6 | 64.7 |
| q05 | False | 132.0 | 108.6 | -23.3 |
| q06 | False | 106.4 | 144.6 | 38.2 |
| q07 | False | 107.4 | 130.7 | 23.3 |
| q08 | False | 170.9 | 113.8 | -57.1 |
| q09 | False | 115.7 | 113.9 | -1.8 |
| q10 | False | 126.1 | 124.9 | -1.2 |
| q11 | False | 100.6 | 101.3 | 0.7 |
| q12 | False | 107.2 | 112.1 | 4.8 |
| q13 | False | 118.9 | 120.9 | 2.0 |
| q14 | False | 102.4 | 130.7 | 28.3 |
| q15 | False | 196.4 | 122.5 | -73.9 |
| q16 | False | 100.3 | 117.5 | 17.2 |
| q17 | False | 106.2 | 129.3 | 23.0 |
| q18 | False | 115.4 | 152.0 | 36.5 |
| q19 | False | 124.8 | 105.6 | -19.2 |
| q20 | False | 178.7 | 101.5 | -77.2 |
| q21 | False | 106.9 | 134.2 | 27.3 |
| q22 | False | 100.2 | 115.5 | 15.3 |
| q23 | False | 153.6 | 110.6 | -43.0 |
| q24 | False | 106.8 | 140.1 | 33.3 |
| q25 | False | 144.1 | 106.4 | -37.8 |
| q26 | False | 121.1 | 157.8 | 36.6 |
| q27 | False | 120.0 | 94.3 | -25.7 |
| q28 | False | 115.1 | 106.4 | -8.7 |
| q29 | False | 108.3 | 124.2 | 15.9 |
| q30 | False | 129.9 | 222.0 | 92.2 |
| q31 | False | 114.9 | 107.9 | -7.0 |
| q32 | False | 130.8 | 108.1 | -22.8 |
| q33 | False | 145.9 | 112.0 | -33.9 |
| q34 | False | 152.4 | 144.3 | -8.1 |
| q35 | False | 106.2 | 104.5 | -1.8 |
| q36 | False | 112.1 | 159.0 | 46.9 |
| q37 | False | 159.1 | 140.0 | -19.1 |
| q38 | False | 130.4 | 130.8 | 0.4 |
| q39 | False | 143.9 | 106.1 | -37.8 |
| q40 | False | 104.0 | 118.6 | 14.6 |
| q41 | False | 125.5 | 105.1 | -20.4 |
| q42 | False | 133.7 | 110.1 | -23.5 |
| q43 | False | 112.8 | 101.2 | -11.6 |
| q44 | False | 129.9 | 126.2 | -3.6 |
| q45 | False | 110.9 | 143.6 | 32.7 |
| q46 | False | 138.0 | 125.7 | -12.4 |
| q47 | False | 149.3 | 107.4 | -41.9 |
| q48 | False | 173.4 | 147.4 | -26.1 |
| q49 | False | 105.6 | 124.1 | 18.5 |

## P3(0.4)

- relative difference: p50 0.1006, p95 0.0411
- observed rerank rate (timed passes): 0.4898
- agreement over 147 rows: 0 decision, 0 top-5 mismatches
- live reranked: n=24 p50 5604.9 p95 6610.4
- live not_reranked: n=25 p50 113.9 p95 142.7
- pass 1 (warm-up): p50 279.1 p95 6439.3
- pass 2 (timed): p50 155.6 p95 6368.0
- pass 3 (timed): p50 268.9 p95 6437.2

| question | rerank | composed ms | live ms | diff ms |
|---|---|---|---|---|
| q01 | False | 124.5 | 104.1 | -20.3 |
| q02 | False | 122.0 | 111.4 | -10.6 |
| q03 | True | 4977.3 | 5298.5 | 321.2 |
| q04 | False | 105.0 | 142.7 | 37.7 |
| q05 | False | 132.1 | 117.4 | -14.7 |
| q06 | False | 106.6 | 134.4 | 27.8 |
| q07 | True | 5745.1 | 6244.3 | 499.2 |
| q08 | False | 171.1 | 111.4 | -59.7 |
| q09 | False | 115.8 | 101.7 | -14.1 |
| q10 | False | 126.2 | 123.5 | -2.6 |
| q11 | False | 100.7 | 97.1 | -3.6 |
| q12 | True | 6231.1 | 6610.4 | 379.3 |
| q13 | False | 119.0 | 132.8 | 13.8 |
| q14 | False | 102.6 | 141.6 | 39.0 |
| q15 | True | 4283.4 | 4650.9 | 367.5 |
| q16 | False | 100.5 | 96.1 | -4.3 |
| q17 | True | 4731.1 | 5227.1 | 496.0 |
| q18 | True | 6138.6 | 5827.7 | -310.9 |
| q19 | False | 124.9 | 113.9 | -11.0 |
| q20 | False | 178.9 | 111.7 | -67.2 |
| q21 | False | 107.1 | 110.9 | 3.8 |
| q22 | False | 100.3 | 113.4 | 13.1 |
| q23 | True | 5401.7 | 6626.1 | 1224.4 |
| q24 | False | 106.9 | 117.3 | 10.3 |
| q25 | True | 5228.4 | 5596.1 | 367.6 |
| q26 | False | 121.2 | 98.6 | -22.6 |
| q27 | False | 120.1 | 96.5 | -23.7 |
| q28 | True | 3864.4 | 3807.2 | -57.2 |
| q29 | True | 5302.6 | 5382.0 | 79.4 |
| q30 | True | 6126.7 | 6062.4 | -64.4 |
| q31 | False | 115.0 | 130.9 | 15.9 |
| q32 | False | 131.0 | 104.5 | -26.4 |
| q33 | True | 4586.1 | 4956.5 | 370.4 |
| q34 | True | 4354.6 | 4445.7 | 91.1 |
| q35 | True | 5047.8 | 5730.3 | 682.6 |
| q36 | True | 6028.6 | 5613.6 | -415.0 |
| q37 | True | 5549.5 | 5874.4 | 324.8 |
| q38 | False | 130.5 | 124.7 | -5.8 |
| q39 | False | 144.0 | 196.9 | 52.9 |
| q40 | True | 5423.2 | 5707.5 | 284.2 |
| q41 | False | 125.7 | 119.7 | -5.9 |
| q42 | True | 5183.1 | 5193.5 | 10.4 |
| q43 | True | 5830.1 | 6309.9 | 479.8 |
| q44 | True | 4154.2 | 4020.6 | -133.6 |
| q45 | True | 5165.9 | 5532.5 | 366.5 |
| q46 | False | 138.1 | 115.3 | -22.8 |
| q47 | True | 5394.4 | 5540.2 | 145.9 |
| q48 | True | 6091.5 | 6119.1 | 27.6 |
| q49 | True | 5693.3 | 6378.3 | 685.0 |
