"""Databricks platformization: governed Delta persistence (Phase 6).

Pure modules (importable and testable without Spark): ``contracts`` (explicit table
contracts + validation), ``identity`` (deterministic record ids and content hashes),
``records`` (persisted row builders from Phase 2-5 outputs), ``sources`` (source
snapshot manifest and hash verification), ``reconcile`` (dataset profiles and
comparison), ``sql`` (Unity Catalog / Delta DDL and MERGE statements), ``review``
(indicator alias review artefact), ``pipeline`` (orchestration).

Spark-dependent code lives only in ``spark_store`` and imports ``pyspark`` lazily.
"""
