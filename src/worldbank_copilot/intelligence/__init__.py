"""Deterministic Gold intelligence layer (Phase 7), built with native Spark on Databricks.

Pure configuration: ``rules`` (attention rules, rating scales) and ``contracts`` (Gold
table contracts). Spark transformations: ``timeline``, ``results``, ``risks``,
``signals``, ``project_360``; shared helpers in ``frames``; invariants in ``checks``;
orchestration in ``pipeline``. No Python UDFs, no predictions, no AI interpretation.
"""
