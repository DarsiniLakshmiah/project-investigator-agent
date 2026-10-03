# Phase 10D bounded synthesis and critique

Status: **IMPLEMENTED_LOCALLY_AWAITING_DATABRICKS_MODEL_VALIDATION**.

The approved path is admitted unexecuted plan -> unchanged deterministic evidence
executor -> EvidenceExecutionReport -> Synthesizer -> Critic -> deterministic finalizer.
The report's InvestigationState and original InvestigationPlan remain unchanged.
SynthesisReport holds the new outputs, extended BudgetLedger, trace events and final
response beside that immutable history. No LangGraph, Investigator or repair loop exists.

The Synthesizer and Critic are model nodes because they each perform one prescribed
inference. They cannot choose tools, SQL, retrieval, projects, transitions or budgets.
The harness owns the fixed sequence and stops on operational or mechanical failure.
One bounded Investigator is deferred to 10E, subject to a separate review gate.

## Contracts and context

CandidateClaim carries claim_id, bounded claim_text, claim_type, provenance_label,
evidence_ids, requirement_ids, source-identity citation references, admitted project_id,
exact temporal_scope and CANDIDATE status. The five existing provenance labels are
reused. ASSERTION, INTERPRETATION and UNCERTAINTY must agree with the provenance.
No self-reported confidence grants permission to publish.

SynthesisOutput contains candidate_claims, insufficient_evidence, limitations and
summary_claim_ids. There is no independently publishable free-text draft. Published
text comes only from individually validated claims; provenance labels accompany it.
Model output is strict JSON, checked against JSON Schema before validated conversion
into detached frozen contracts. Duplicate JSON keys, extra fields, wrong scalar types,
unbounded output and malformed schemas fail closed. Install requirements-phase10d.txt
for the JSON Schema validator; existing Phase 9/10C dependencies are unchanged.

Context construction revalidates the 10C report, package fingerprint, admitted scope,
requirement membership and nested owner metadata. It retains whole evidence references
and their unchanged citations/source identities, deduplicates through the existing
evidence index, and orders requirements/evidence by identity. The question and source
content are explicitly untrusted data. Credentials/clients/configuration are never
included; credential-shaped fields and recognizable bearer/token values are refused.
Arbitrary sensitive prose cannot be classified reliably at this boundary: trusted
upstream evidence selection and project authorization remain necessary.

Whole-reference selection uses a conservative serialized UTF-8 byte budget, including
recorded omissions. These are policy accounting units, not actual tokenizer estimates.
Requests additionally bound instructions/schema/context; oversize metadata or candidate
output fails closed. Missing/omitted required evidence prevents successful publication.
No model summarizes context. Prompt instructions preserve uncertainty, distinguish
signals from Bank judgments, prohibit failure predictions and atomic-snapshot claims,
and request structured conclusions without chain-of-thought.

## Validation and publication

Mechanical validators check unique claim IDs; known, included, project-scoped evidence;
valid linked requirements; exact source-identity references; citations for every cited
item; provenance compatibility; admitted project and exact temporal scope; output bounds.
Unknown provenance remains uncertainty. Signal/document provenance cannot become FACT.
Interpretations must be labeled and backed by supplied references. A narrow textual
policy guard catches explicit failure-prediction/atomicity/Bank-judgment phrases; it is
not a general natural-language entailment engine.

CriticOutput contains one finding per claim, a constrained CriticCode, approved evidence
IDs and concise rationale. Codes are SUPPORTED, UNSUPPORTED, CONTRADICTED,
CITATION_INVALID, PROVENANCE_INVALID, PROJECT_SCOPE_VIOLATION,
TEMPORAL_SCOPE_VIOLATION, OVERCLAIMED and INSUFFICIENT_EVIDENCE. It cannot rewrite
claims, add evidence, change scope or publish. Refuting findings may cite other supplied
evidence; SUPPORTED must cite exactly the claim's evidence. Missing/duplicate findings
and invented IDs fail closed.

The finalizer repeats mechanical checks and checks critic completeness. Invalid output
or failed/missing critic yields FAIL_CLOSED. Any non-supported finding rejects the
whole draft with REJECT_UNSUPPORTED. Unknown claims, explicit insufficiency, empty
output/evidence or uncovered required requirements yield INSUFFICIENT_EVIDENCE.
Only mechanically valid, supported claims publish. The current 10C cross-source
atomicity limitation always applies, so eligible answers use PUBLISH_WITH_LIMITATIONS.
No partial filtering hides a rejected required claim. PUBLISH is reserved in the enum.

Semantic support is MODEL_ASSESSED, never mechanically proved. A critic can miss a
semantic error. Deterministic validation catches mechanical violations even if the
critic says SUPPORTED; it cannot establish arbitrary factual entailment. Capability
validation and human review remain gates before production promotion.

Each node reserves one model call and a bounded token allowance in the existing ledger
before invoking its adapter. No reservation is refunded after a failed call. Deadlines,
aggregate budgets and usage overruns fail closed. No repair cycle or retry is reserved.
Production run_nodes requires the unchanged approved live cost configuration before
invocation; offline fake-adapter tests use an explicit trusted offline flag. The
separate synthetic capability experiment has fixed call bounds and unavailable billing.
Observed usage may be unavailable. No prices, cost ceiling or zero-billing claim is
invented; any existing configured monetary limits remain unchanged. This checkpoint
does not authorize production billing/pricing policy or live agent execution.

Failure categories distinguish model unavailability/timeout, invalid output/schema,
evidence/project/provenance/citation/temporal/requirement violations, context/output
budgets, critic rejection, insufficiency and internal integrity errors. Error details do
not quote raw exceptions or model-generated text.

ModelEvent prepares metadata for later tracing: correlation/project/plan/package identity,
prompt/schema version, role/model identity, included evidence/claim IDs, critic codes,
latency, numeric token usage and typed errors. Full prompts, credentials, raw HTTP
responses and chain-of-thought are not traced. No MLflow instrumentation is added.

## User-run capability experiment

The adapter accepts a bounded ModelRequest and returns final text and numeric usage.
Domain logic is independent of HTTP. DatabricksModelAdapter reuses the existing 9D
single-attempt chat transport without changing it; it sends strict structured-output
schema and no tools. Non-stop completion, refusal or tool invocation fails closed.
Only text parts are read; reasoning content is never retained. Endpoint identity and
an optional explicit reasoning setting belong to trusted wiring, not model output.

The 11 preregistered cases in evaluation/phase10d_model_cases.json cover:

- S-supported: positive synthesis schema/reference/scope baseline.
- S-signal: preserve deterministic signal provenance.
- S-unknown: missing value and abstention.
- S-injection: refuse untrusted requests to fabricate IDs or change project.
- C-supported: accept the properly supported claim.
- C-unsupported: reject an unsupported budget statement with valid IDs.
- C-fabricated: reject nonexistent evidence.
- C-project: reject a different project.
- C-provenance: reject signal promotion to FACT.
- C-overclaim: detect failure prediction.
- C-temporal: detect changed temporal scope.

Supported synthesis, unknown synthesis, supported critique and unsupported critique
repeat three times. All other cases run once: 19 calls per endpoint, at most two
explicitly supplied endpoint candidates, no retries. Contract stability compares
flags, evidence/requirement links, provenance/types, critic decisions and acceptance
checks, not byte-identical prose. The synthesis semantic check is a narrow, documented
literal fixture check (delay present, unrelated budget/failure assertions absent), not
a general quality score or proof of entailment. These synthetic cases do not replace a
future end-to-end project-answer evaluation or human semantic review.

The repository documents earlier GPT-OSS and unavailable nano endpoints, but those
results were for routing. No currently available endpoint is established offline and
none is selected here. The user supplies one or two approved endpoint names. Strict
schema/settings support itself is a capability gate; errors are preserved, not hidden
through an unconstrained-output fallback. Some models may require additional settings;
trusted custom adapter wiring can specify reasoning_effort. The thin notebook uses the
standard settings only and makes no claim they work before the real experiment.

Artifacts record environment/revision, source locks, schedule, endpoint/model identity,
role, prompt/schema versions, settings, attempt count, operational/schema/reference/
scope/provenance checks, stability, latency, token usage and billed-cost UNAVAILABLE.
No model is promoted automatically. Offline artifacts explicitly say OFFLINE_STAND_INS.

## Integrity, persistence and next gate

The thin 09_phase10d_model_validation.py wrapper uses the unchanged accepted AST
canonicalizer; exact notebook byte equality is not required. Ordered executable
statements and directives remain protected. The 10C-specific approved import/widget
commute is not expanded to admit new 10D reorderings: unrecognized execution changes
still fail closed. Substantive 10D files/cases/dependencies have exact normalized-LF
hashes. prepare() first verifies unchanged Phase 9/10C contracts, then the 10D lock,
case schedule, runtime module association and semantic wrapper identity.

A unique attempt is exclusively reserved before preflight. Checkpoints, final JSON and
SHA/byte receipts are new files only; failures/crashes are preserved. Readback and
receipt validation gate completion. This is sequential FUSE-compatible persistence,
not a distributed atomic-publication claim. Trustworthy Git metadata must match the
reviewed SHA and locked substantive files must be committed clean; ambiguous metadata
records the declared reviewed revision separately and still requires content locks.

Next: review implementation/cases/lock; commit and push yourself; sync the Databricks
Git folder; use a fresh Python session; choose available approved endpoints; run the
thin wrapper with commit_sha, a new 10d run_id and endpoints. Preserve the final JSON,
its completion receipt and all failed sidecars for review. No capability result is
claimed locally, no production model is chosen and Phase 10E remains unstarted.
