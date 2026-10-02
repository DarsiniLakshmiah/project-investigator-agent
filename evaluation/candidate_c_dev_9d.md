# Candidate C DEV evaluation - dev1

**Outcome: REJECTED** (precedence: INVALID > REJECTED_SAFETY > INCONCLUSIVE_OPERATIONAL > REJECTED > ACCEPTED_FOR_NEXT_STAGE)

TEST was not evaluated. C2 is shadow diagnostic evidence only.

## Findings
- INVALID: none
- REJECTED_SAFETY: none
- INCONCLUSIVE_OPERATIONAL: none
- REJECTED: r015: C1 intent CHANGE_INVESTIGATION != EXPLANATION, r015: C1 final route INVESTIGATION != DOCUMENT, r015: wrong executable route INVESTIGATION, r015: C1 repeats ['EXPLANATION', 'CHANGE_INVESTIGATION', 'CHANGE_INVESTIGATION', 'CHANGE_INVESTIGATION', 'CHANGE_INVESTIGATION'] vs main CHANGE_INVESTIGATION, hybrid 25/29 != required 26/29, hybrid changes that are not corrections: [('r015', 'changed_still_wrong'), ('r048', 'correction')]

## C1 (production boundary)
| case | expected | predicted | final route | intent ok | route ok | repeats stable |
|---|---|---|---|---|---|---|
| r015 | EXPLANATION/DOCUMENT | CHANGE_INVESTIGATION | INVESTIGATION | False | False | False |
| r048 | DOCUMENT_CONTENT/DOCUMENT | DOCUMENT_CONTENT | DOCUMENT | True | True | True |

## Hybrid vs Candidate A
Candidate A 24/29; hybrid 25/29 (delta +1); changes: [{'case_id': 'r015', 'effect': 'changed_still_wrong'}, {'case_id': 'r048', 'effect': 'correction'}]

## C2 shadow diagnostics (human review required)
intent accuracy 0.5455, route accuracy 0.6818, abstain rate 0.0, matrix {'both_correct': 13, 'deterministic_only': 4, 'gpt_only': 2, 'both_wrong': 3}, patterns [{'expected': 'EXPLANATION', 'predicted': 'CHANGE_INVESTIGATION', 'count': 3}]

Latency: {'inference_p50_s': 0.3244, 'inference_p95_s': 1.5515, 'note': 'inference only; the 5 s gap is request pacing, not inference time'}. Counts: {'dev_scheduled_calls': 44, 'c1_calls': 12, 'calls_made': 44, 'operational_failures': 0, 'contract_failures': 0, 'operational_failure_rate': 0.0}.
