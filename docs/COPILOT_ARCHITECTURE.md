# Copilot architecture

Status: implemented locally; real Databricks validation and human acceptance pending. Phase 7/8/9/10C are accepted historical components. Phase 10D is unaccepted. Preserve 10d1 (preflight failure), 10d2 (interrupted reservation), 10d3/10d4 (preflight identity mismatch) and 10d5 (19 failed transport requests). The real diagnosis established unsupported JSON Schema pattern at HTTP 400, before model generation; see [the compatibility revision](PHASE10D_TRANSPORT_COMPATIBILITY.md). 10d6 (@3, GPT-OSS-20b) executed 19/19 calls with 14 PASS and remains a valid historical model-capability FAIL; the deterministic UNKNOWN-evidence provenance gap it exposed is closed in [lock @4](PHASE10D_UNKNOWN_PROVENANCE.md). No synthesis/critic quality acceptance follows.

## Audit, reuse and request lifecycle

No accepted component was rewritten. The existing Phase 9 deterministic-first RoutingService, contracts and entity index handle query understanding/routing. Existing Phase 8 fixed/hybrid/cross_encoder/k50 retrieval supplies metadata-filtered lexical/vector search, reranking and references. Governed Gold/Silver readers remain Spark-based. Existing 10B policy, 10C admission/template planning/evidence execution/assembly and 10D adapter/synthesis/validation/critic/finalizer are reused unchanged.

Streamlit App -> strict question/project/session request -> trusted single-notebook Job -> Application.answer_question -> Phase 9 router. STRUCTURED and DOCUMENT return source evidence without generated explanations. Unsupported direct document date filtering clarifies. INVESTIGATION executes the accepted initial evidence plan; unresolved requirements can trigger one proposal-only Investigator call and at most one additional trusted evidence operation. Existing synthesis, mechanical validators, tool-free critic and deterministic finalizer then execute. Generated claims retain the existing mandatory critic gate.

The Investigator alone is agentic: its strict Decision selects SEARCH_DOCUMENTS, GET_TIMELINE_EVIDENCE or STOP. The deterministic harness verifies unresolved target, derives trusted project/time/tool arguments, applies existing budgets and executes only the new requirement. Invalid output, errors and exhausted allowance stop safely. No recursive loop, autonomous SQL or unrestricted tool executes. Original evidence is immutable and initial operations are not repeated.

Synthesis and criticism are specialized model stages. Mechanical checks prove ownership, provenance, reference existence, schema and budget compliance. The critic judges semantic relevance, sufficiency, grounding and overclaiming. Reference existence alone cannot prove semantic support.

## Engineering layers

| Layer | Responsibility |
|---|---|
| Prompt | Bounded role instructions and strict output schemas; no chain-of-thought requested. |
| Context | Project/time-specific evidence and compact unresolved state. Explicit deterministic application projection retains requirements/conflict references and declares omitted references, without modifying original evidence or historical bounds. |
| Harness | Allowlisted actions/projects, trusted temporal arguments, schema/citation validation and existing conservative budgets. |
| Graph | Fixed route/evidence/optional investigation/synthesis/validation/critic/finalization paths. |
| Loop | One proposal, at most one extra operation, one synthesis and one critic; no retry loop. |
| Retrieval | Accepted hybrid search, metadata filtering and cross-encoder reranking. |
| Evaluation | Preregistered A/B/C protocol; mechanical invariants and actual runtime measurements separated from human semantic review. |
| Observability | Real MLflow root/stage/model spans and safe structured attributes. |

## Guardrails, sessions and observability

Each request authorizes only its selected registered project. Credential-shaped input is rejected before submission. Models cannot add project/time/tool controls; retrieved text remains untrusted evidence. Existing prediction/citation protections remain. Session continuity stores only the previous answer in Streamlit memory. Source follow-ups show an explicit previous snapshot; new questions fetch fresh evidence. No persistent semantic memory or cross-project cache is added.

Trace attributes include request/session/project, UTC timestamp, configuration version, route, durations, model identities/tokens when supplied, model/retrieval/action counts, required-root coverage, critic codes, evidence IDs and outcome. This integration does not log raw prompts/documents, credentials, arbitrary exception messages or chain-of-thought. Workspace-managed SDK/platform logging requires its own governance.

## Evaluation and limitations

A is deterministic evidence; B adds synthesis/critic; C permits the Investigator. Six fixed cases across three modes schedule 18 responses including unsupported-project refusal. Mechanical requirement coverage means roots with references, not semantic sufficiency. Artifacts always retain NOT_ACCEPTED/HUMAN_REVIEW_REQUIRED. Keep B until reviewed real quality improvement justifies C. If C never invokes investigation, no benefit is demonstrated.

Notebook 13 makes real proposal calls on explicitly synthetic empty-evidence contexts without executing tools. It proves contract validity only. Adversarial retrieved-content, insufficiency, temporal and critic-quality behavior require real review alongside the preserved 10D suite and comparison responses. Local mocks are neither model results nor real MLflow traces.

This is a controlled demonstration with a shared backend identity and trusted static project allowlist, not enterprise per-user authorization. Job startup/runtime initialization add latency. Unsupported date filtering clarifies. Context projection can reduce coverage and reports omissions. Costs use configured conservative bounds, not fabricated bills. No agent swarm, LangGraph, NeMo Guardrails, predictive model, durable memory or autonomous deployment was added because these are unnecessary for the bounded design.

Interview description: a bounded agentic RAG architecture with deterministic orchestration, specialized LLM roles, and one tool-selecting Investigator agent. More agents would add latency, cost, nondeterminism and debugging complexity without demonstrated benefit.
