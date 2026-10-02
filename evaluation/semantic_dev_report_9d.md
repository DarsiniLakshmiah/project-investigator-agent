# Phase 9D — bounded semantic routing: development report (first stop)

**Development split only.** The 51 frozen TEST cases have not been used by any classifier, embedding, threshold or selection step (`test_evaluated: false`; dataset sha256 `83c5f9123ecef96c…`).

## Provisional decision

- **Selected:** A (rules-only, semantic fallback disabled) (status PROVISIONAL)
- **Reason:** no B-lexical configuration met the pre-registered development rule (route precision >= 0.9 at coverage >= 0.3); abstention -> CLARIFY is preferred to a wrong route
- **A:** rules-only 9B.2 (baseline, always available)
- **B-lexical:** evaluated on DEVELOPMENT - REJECTED
- **B-qwen:** PENDING - run notebooks/08_semantic_routing_dev.py (DEVELOPMENT only) in Databricks against the validated Qwen endpoint
- **C:** PROPOSED, not implemented - awaiting approval and an endpoint probe
- **Freeze rule:** becomes FROZEN only after the B-qwen development run is reviewed; TEST stays blocked until then (assert_split_allowed)

## B-lexical: leave-one-family-out on 22 development examples

Embedder `lexical-hash-v1(dims=262144,char=(3, 4))`. Majority-route baseline: always STRUCTURED = 0.5 route accuracy.

### Unthresholded configurations (no abstention)

| config | answered | route accuracy | intent accuracy |
|---|---|---|---|
| k=1,w=similarity,sim>=0.00,share>=0.00 | 22/22 | 0.227 | 0.045 |
| k=1,w=uniform,sim>=0.00,share>=0.00 | 22/22 | 0.227 | 0.045 |
| k=3,w=similarity,sim>=0.00,share>=0.00 | 22/22 | 0.227 | 0.045 |
| k=3,w=uniform,sim>=0.00,share>=0.00 | 22/22 | 0.273 | 0.045 |
| k=5,w=similarity,sim>=0.00,share>=0.00 | 22/22 | 0.227 | 0.045 |
| k=5,w=uniform,sim>=0.00,share>=0.00 | 22/22 | 0.318 | 0.091 |

### Highest route precision among answered predictions (any coverage)

| config | answered | coverage | route precision | intent precision |
|---|---|---|---|---|
| k=3,w=uniform,sim>=0.50,share>=0.50 | 1/22 | 0.045 | 1.0 | 0.0 |
| k=3,w=uniform,sim>=0.20,share>=0.50 | 4/22 | 0.182 | 0.5 | 0.0 |
| k=3,w=uniform,sim>=0.40,share>=0.50 | 2/22 | 0.091 | 0.5 | 0.0 |
| k=3,w=uniform,sim>=0.00,share>=0.50 | 5/22 | 0.227 | 0.4 | 0.0 |
| k=3,w=uniform,sim>=0.10,share>=0.50 | 5/22 | 0.227 | 0.4 | 0.0 |
| k=5,w=uniform,sim>=0.10,share>=0.00 | 21/22 | 0.955 | 0.333 | 0.095 |

Selection rule: route precision >= 0.9 at coverage >= 0.3. Configurations evaluated: 108. Qualifying: none.

### Coverage vs accuracy (base config k=3,w=similarity,sim>=0.00,share>=0.00, ranked by confidence)

| kept | coverage | min confidence | route accuracy | intent accuracy |
|---|---|---|---|---|
| 1 | 0.045 | 0.4305 | 0.0 | 0.0 |
| 2 | 0.091 | 0.3048 | 0.0 | 0.0 |
| 3 | 0.136 | 0.2925 | 0.0 | 0.0 |
| 4 | 0.182 | 0.2865 | 0.0 | 0.0 |
| 5 | 0.227 | 0.2769 | 0.0 | 0.0 |
| 6 | 0.273 | 0.2741 | 0.0 | 0.0 |
| 7 | 0.318 | 0.2602 | 0.0 | 0.0 |
| 8 | 0.364 | 0.2448 | 0.125 | 0.0 |
| 9 | 0.409 | 0.1878 | 0.222 | 0.0 |
| 10 | 0.455 | 0.1703 | 0.2 | 0.0 |
| 11 | 0.5 | 0.1603 | 0.182 | 0.0 |
| 12 | 0.545 | 0.1593 | 0.25 | 0.0 |
| 13 | 0.591 | 0.1414 | 0.231 | 0.0 |
| 14 | 0.636 | 0.1414 | 0.214 | 0.0 |
| 15 | 0.682 | 0.1334 | 0.2 | 0.0 |
| 16 | 0.727 | 0.1131 | 0.188 | 0.0 |
| 17 | 0.773 | 0.1 | 0.235 | 0.0 |
| 18 | 0.818 | 0.0976 | 0.222 | 0.0 |
| 19 | 0.864 | 0.0966 | 0.211 | 0.0 |
| 20 | 0.909 | 0.0631 | 0.2 | 0.0 |
| 21 | 0.955 | 0.0631 | 0.238 | 0.048 |
| 22 | 1.0 | 0.0139 | 0.227 | 0.045 |

### Per-example leave-one-family-out predictions (base config)

| case | expected route | expected intent | predicted route | predicted intent | top similarity | confidence |
|---|---|---|---|---|---|---|
| r002 | STRUCTURED | RESULTS_PROGRESS | STRUCTURED | PROJECT_OVERVIEW | 0.241 | 0.1593 |
| r006 | STRUCTURED | RATING_HISTORY | INVESTIGATION | CHANGE_INVESTIGATION | 0.336 | 0.1603 |
| r012 | STRUCTURED | FINANCIAL_STATUS | DOCUMENT | EXPLANATION | 0.401 | 0.1414 |
| r014 | DOCUMENT | DOCUMENT_CONTENT | STRUCTURED | RATING_HISTORY | 0.179 | 0.1334 |
| r015 | DOCUMENT | EXPLANATION | STRUCTURED | PROJECT_OVERVIEW | 0.173 | 0.0966 |
| r016 | DOCUMENT | DOCUMENT_CONTENT | INVESTIGATION | CHANGE_INVESTIGATION | 0.641 | 0.2925 |
| r017 | DOCUMENT | EXPLANATION | STRUCTURED | FINANCIAL_STATUS | 0.401 | 0.1414 |
| r020 | DOCUMENT | DOCUMENT_CONTENT | STRUCTURED | TIMELINE_EVENTS | 0.373 | 0.1703 |
| r025 | DOCUMENT | DOCUMENT_CONTENT | STRUCTURED | RISKS | 0.236 | 0.0976 |
| r029 | STRUCTURED | PROJECT_OVERVIEW | DOCUMENT | EXPLANATION | 0.548 | 0.2602 |
| r030 | STRUCTURED | PROJECT_OVERVIEW | STRUCTURED | RESULTS_PROGRESS | 0.241 | 0.1 |
| r034 | STRUCTURED | RATING_HISTORY | STRUCTURED | RISKS | 0.441 | 0.2448 |
| r037 | STRUCTURED | ATTENTION | INVESTIGATION | CHANGE_INVESTIGATION | 0.143 | 0.0631 |
| r042 | STRUCTURED | TIMELINE_EVENTS | DOCUMENT | DOCUMENT_CONTENT | 0.373 | 0.2741 |
| r044 | STRUCTURED | RISKS | DOCUMENT | DOCUMENT_CONTENT | 0.236 | 0.1131 |
| r045 | STRUCTURED | RISKS | STRUCTURED | RATING_HISTORY | 0.441 | 0.1878 |
| r048 | DOCUMENT | DOCUMENT_CONTENT | STRUCTURED | RATING_HISTORY | 0.036 | 0.0139 |
| r050 | DOCUMENT | EXPLANATION | STRUCTURED | PROJECT_OVERVIEW | 0.548 | 0.4305 |
| r052 | INVESTIGATION | CHANGE_INVESTIGATION | STRUCTURED | FINANCIAL_STATUS | 0.45 | 0.2865 |
| r055 | INVESTIGATION | CHANGE_INVESTIGATION | DOCUMENT | DOCUMENT_CONTENT | 0.641 | 0.3048 |
| r056 | INVESTIGATION | CHANGE_INVESTIGATION | INVESTIGATION | CHANGE_INVESTIGATION | 0.143 | 0.0631 |
| r060 | STRUCTURED | FINANCIAL_STATUS | INVESTIGATION | CHANGE_INVESTIGATION | 0.45 | 0.2769 |

### End-to-end on DEVELOPMENT (29 cases)

- Candidate A, rules only (semantic-needed -> CLARIFY): route accuracy 24/29; semantic-needed 2/29.
- Hypothetical fallback with the rejected base config: route accuracy 24/29; semantic invocations 2/29; semantic correct 0/2; abstained 0/2.
- Fallback-eligible development cases: r015, r048 (denominator 2 - too small to support any accuracy claim).
- Classifier latency (in-process): p50 0.241 ms, p95 0.422 ms (n=22).

## False deterministic resolutions and other 9B.2 mismatches

Read from the recorded 9C rule baseline (no classifier involved). DETERMINISTIC_FALSE_RESOLUTION = the rules confidently chose a wrong executing route; those are harness/rule problems, not classifier territory, and are NOT repaired here.

| case | split | human | 9B.2 | type | category | evidence | recommendation (future harness revision) |
|---|---|---|---|---|---|---|---|
| r002 | dev | STRUCTURED | STRUCTURED / INTENT_RESULTS_PROGRESS | TEMPORAL_DIFFERS | deterministic lexical-rule error | 'current value' is read as an explicit LATEST time (the route is right). | Treat 'current value' as a results field name, not a time word. |
| r003 | test | STRUCTURED | DOCUMENT / INTENT_DOCUMENT_CONTENT | DETERMINISTIC_FALSE_RESOLUTION | deterministic lexical-rule error | DOCUMENT_ACCORDING_TO fires on 'according to ISR 24', which only gives source context for an extracted value (human rule 2). | Subject x source decision table: a document reference plus an extracted structured subject (ISR disbursement, ratings) stays STRUCTURED. |
| r007 | test | CLARIFY | CLARIFY / TIME_REQUIRED | TEMPORAL_DIFFERS | intentional conservative behavior | Route and reason agree (CLARIFY TIME_REQUIRED); only the recorded time kind differs (defaulted LATEST vs labelled NONE). | None needed; optionally record no default when TIME_REQUIRED fires. |
| r008 | test | DOCUMENT | STRUCTURED / INTENT_RISKS | DETERMINISTIC_FALSE_RESOLUTION | semantic ambiguity | 'main environmental risks' needs the document's own prioritisation (rule 1); the rules only see the subject 'risks'. | Salience qualifiers (main/key/principal) on risks -> DOCUMENT, or leave to a validated semantic layer. |
| r011 | test | STRUCTURED | DOCUMENT / INTENT_DOCUMENT_CONTENT | DETERMINISTIC_FALSE_RESOLUTION | deterministic lexical-rule error | DOCUMENT_IN_NAMED_DOCUMENT fires on 'identified in the fiduciary systems assessment'; the findings are extracted in the risk register (rule 2). | Same table: assessment findings named by their source document -> RISKS with a source_document_type filter. |
| r025 | dev | DOCUMENT | DOCUMENT / INTENT_DOCUMENT_CONTENT | TEMPORAL_DIFFERS | deterministic lexical-rule error | 'trends' is read as HISTORY although it is the subject matter (route right). | Remove 'trends' from the HISTORY lexicon or require a time context. |
| r027 | test | INVESTIGATION | DOCUMENT / INTENT_EXPLANATION | DETERMINISTIC_FALSE_RESOLUTION | deterministic lexical-rule error | Explanatory + decision subject -> EXPLANATION, but 'which restructuring ... and why' must first establish the event (rule 4). | Encode rule 4: 'which <decision/event> ... why' -> INVESTIGATION. |
| r033 | test | STRUCTURED | CLARIFY / TEMPORAL_NOT_SUPPORTED | ROUTE_DIFFERS: 9B CLARIFY vs human STRUCTURED | tool capability limitation | get_rating_history has no date filter, so a YEAR scope is CLARIFY (TEMPORAL_NOT_SUPPORTED) - intentionally conservative. | Add an ISR-date filter to the rating-history tool (ISR dates are governed). |
| r040 | test | STRUCTURED | STRUCTURED / INTENT_TIMELINE_EVENTS | TOOLS_OR_ARGUMENTS_DIFFER | deterministic lexical-rule error | Event types are taken from the whole question, so the anchor phrase 'since the 2021 restructuring' adds RESTRUCTURING to the requested CLOSING_DATE_CHANGE. | Exclude the temporal-anchor span from event-type extraction. |
| r044 | dev | STRUCTURED | DOCUMENT / INTENT_DOCUMENT_CONTENT | DETERMINISTIC_FALSE_RESOLUTION | deterministic lexical-rule error | 'from the ESSA' triggers document scope for a request to LIST extracted findings (rule 2). | Same table; 'list' + extracted findings -> RISKS. |
| r053 | test | INVESTIGATION | INVESTIGATION / INTENT_CHANGE_INVESTIGATION | TEMPORAL_DIFFERS | semantic ambiguity | 'during appraisal' qualifies the risks, not the period of the investigation (labelled HISTORY implicit). | Distinguish qualifier vs scope for appraisal wording inside investigations. |
| r055 | dev | INVESTIGATION | CLARIFY / AMBIGUOUS_TIME | ROUTE_DIFFERS: 9B CLARIFY vs human INVESTIGATION | temporal capability limitation | The event-anchor grammar reads 'before the restructuring, additional financing and cancellation events' as one ambiguous anchor -> CLARIFY. | Event-set anchors ('before the X, Y and Z events') -> the history of those event types, not a single anchor. |
| r056 | dev | INVESTIGATION | INVESTIGATION / INTENT_CHANGE_INVESTIGATION | TEMPORAL_DIFFERS | intentional conservative behavior | Investigation default time HISTORY vs labelled LATEST; the plan already asks for CURRENT signals. | Per-investigation-kind default time (attention -> LATEST). |
| r060 | dev | STRUCTURED | CLARIFY / AMBIGUOUS_TIME | ROUTE_DIFFERS: 9B CLARIFY vs human STRUCTURED | temporal capability limitation | Relative periods are UNRESOLVED in 9B.2; the reviewed contract resolves them against the loan-statement as-of date (2026-08-31). The finance tool also has no date interval. | Governed as-of resolution per intent (finance: statement snapshot) and a finance-history capability (ISR-printed values by date). |

