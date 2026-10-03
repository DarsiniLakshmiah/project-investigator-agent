# Phase 10D pre-acceptance protocol revision 2

10d1 used the preregistered databricks_wrapper_ast@2 protocol at reviewed
revision 0228b4dfce5f10c9fefe0c7a363905cba7d66a1a. Actual Databricks diagnostics
isolated PHASE10D_CONTENT_OR_PROTOCOL_MISMATCH to notebook semantic identity.
All other lock fields matched. The inherited Phase 10C normalization recognized
only its own import and two widgets; the Phase 10D import followed three widgets.
The observed @2 digest was 08b8e65b44ed4a14a74889a417613cfa17d54006c4c49a41b211ab9f9751b8ec.
10d1 failed preflight before capability evaluation. It supplies no model capability
conclusion and is not a model failure. Preserve its artifacts and receipts exactly.

The original preregistration is archived byte-for-byte as
`evaluation/phase10d_model_lock_v1.json`; its LF-normalized digest is
`ef9f5f7c59de938389933d9837252df2e08b2bf001f68a40423af73a999f43e3`.
The revised active lock uses `phase10d_capability_lock@2` and records that digest
in `supersedes_lock_sha256_lf`. Historical Phase 9/10C locks and code are unchanged.

The 10D-owned `phase10d_notebook_v3.py` reuses the unchanged accepted parser and
canonical JSON payload, and labels the identity `databricks_wrapper_ast@3`.
Only the exact phase10d_models.run_databricks_validation import at either end of
the adjacent ordered commit_sha, run_id, endpoints widget block immediately after
%run ./_bootstrap is normalized to import-first. Each widget must be a direct
three-positional-string-argument declaration with no keywords. Adjoining extra
widgets, unknown imports, reordered/missing widgets, effects inside arguments,
and intervening statements do not normalize. All widget argument ASTs are retained.
Unknown or altered programs remain identity-sensitive and fail the frozen comparison.
The payload SHA remains 754e42ac71c378df721b258b23a2c9b9b1d4aef7c8c97ef2da1d942ad957c349.

Only the 10D harness substantive hash changes: it selects @3, records protocol
lineage and locks two new files (the canonicalizer and its focused regressions).
The case hash, existing tests, model adapter, prompts, schemas, fixtures, requirements,
bootstrap, Phase 10C reference hash, and all capability limits remain unchanged.
This is infrastructure/protocol correctness before acceptance, not experiment tuning.

Local preparation and offline tests do not establish Databricks acceptance.
After user review, commit and synchronization, the user must manually perform the
next validation using a NEW immutable run ID (10d2 if unused). Do not reuse 10d1.
Codex must not create/reserve that attempt or invoke any endpoint.
The diagnostic helper checks the revised lock and dispatches to each phase's
identity implementation; its default reviewed SHA still identifies the historical
reviewed checkout and must not be treated as the future commit SHA.
