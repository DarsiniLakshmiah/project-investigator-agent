"""Databricks construction of ``Copilot`` from the existing governed components.

Composes the same constructors as the accepted live wiring (Phase 9 protocol gate,
Spark table reader, Phase 8 hybrid retrieval with the accepted Vector Search index and
CrossEncoder profile, governed tool catalog, DatabricksModelAdapter wrapped only to record
sanitized failure reason codes). Nothing is
reimplemented here; this module only connects them for the prototype runtime.
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.common.dependency_health import check_environment
from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.copilot.config import CopilotConfig, load_copilot_config
from worldbank_copilot.copilot.model_diagnostics import DiagnosedModelAdapter
from worldbank_copilot.copilot.service import Copilot
from worldbank_copilot.ingestion.documents import load_document_manifest
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.contract import DocumentRetrieval, build_document_search
from worldbank_copilot.retrieval.embeddings import embedding_provider
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
from worldbank_copilot.routing.config import load_routing_config
from worldbank_copilot.routing.entities import EntityIndex
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.reader import SparkTableReader
from worldbank_copilot.tools.registry import TOOL_SPECS
from worldbank_copilot.validation import phase9_contract

# Pinned runtime dependencies: Phase 8/9 retrieval stack plus 10D output validation.
REQUIREMENT_FILES = (*phase9_contract.REQUIREMENTS, "requirements-phase10d.txt")


def build_copilot(
    spark: Any, settings: Any, config: CopilotConfig | None = None, *, tracing: bool = True
) -> Copilot:
    """Wire the prototype runtime against live Databricks data, index and endpoints."""
    if settings.environment.value != "databricks":
        raise ConfigurationError("build_copilot requires the Databricks environment")
    config = config or load_copilot_config(settings.config_dir)
    health = check_environment(settings.repo_root, REQUIREMENT_FILES, settings.config_dir)
    if not health.ok:
        raise ConfigurationError("DEPENDENCY_HEALTH_FAILED\n" + health.format())
    # Frozen Phase 9 gate: verifies pinned sources and yields the accepted retrieval profile.
    protocol = phase9_contract.prepare_protocol(settings.repo_root, dependency_ok=health.ok)
    retrieval_settings, profile = protocol.settings, protocol.profile
    registry = load_project_registry(settings.config_dir)
    if set(config.allowed_projects) - set(registry.project_ids):
        raise ConfigurationError("allowed_projects must be registered projects")

    store = ChunkStore.from_table(spark, rp.names(settings, retrieval_settings)["chunks"])
    dense = rp.vector_index(settings, retrieval_settings)
    _require_accepted_index(dense.describe(), len(store.rows), protocol)
    retriever = Retriever(
        store,
        retrieval_settings,
        registry.project_ids,
        embedding_provider(retrieval_settings.embeddings),
        dense,
    )
    search = build_document_search(
        profile, retriever, CrossEncoderReranker(retrieval_settings.retrieval.reranker)
    )
    tools = ToolExecutor(TOOL_SPECS)
    rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)

    def context(request_id: str) -> ToolContext:
        reader = SparkTableReader.for_settings(spark, settings)
        return ToolContext(reader, registry, scales, rules, request_id)

    routing = load_routing_config(settings.config_dir)
    entities = EntityIndex.build(
        registry, load_document_manifest(settings.config_dir / "document_manifest.yaml"), routing
    )
    router = RoutingService(routing, entities, tools, context, documents=search, semantic=None)
    if tracing:
        require_tracing_api()
        if config.mlflow_experiment:
            import mlflow

            mlflow.set_experiment(config.mlflow_experiment)
    models = config.models
    return Copilot(
        router=router,
        documents=DocumentRetrieval(profile, search, tools),
        config=config,
        config_dir=settings.config_dir,
        investigator=DiagnosedModelAdapter(models.investigator_endpoint),
        synthesizer=DiagnosedModelAdapter(models.synthesizer_endpoint),
        critic=DiagnosedModelAdapter(models.critic_endpoint) if models.critic_enabled else None,
        mlflow_enabled=tracing,
    )


def require_tracing_api() -> None:
    """Fail at build time if the runtime MLflow lacks what ``Recorder`` uses.

    Tracing uses the runtime-provided (protected) ``mlflow-skinny``; nothing installs MLflow.
    """
    import mlflow
    from mlflow.entities import LiveSpan

    missing = [name for name in ("start_span", "set_experiment") if not hasattr(mlflow, name)]
    missing += [f"LiveSpan.{n}" for n in ("trace_id", "set_attributes") if not hasattr(LiveSpan, n)]
    if missing:
        raise ConfigurationError(
            f"MLFLOW_TRACING_API_UNAVAILABLE (mlflow {getattr(mlflow, '__version__', '?')}): "
            + ", ".join(missing)
        )


def _require_accepted_index(description: dict, corpus_rows: int, protocol) -> None:
    """Same identity check as the accepted Phase 10C live validation."""
    status = description.get("status") or {}
    expected = protocol.manifest["retrieval"]["index"]
    if (
        description.get("endpoint_name") != protocol.profile.vector_search_endpoint
        or description.get("name") != phase9_contract.INDEX_NAME
        or status.get("indexed_row_count") != expected["index_rows"]
        or corpus_rows != expected["corpus_rows"]
        or status.get("ready") is not True
    ):
        raise ConfigurationError("LIVE_INDEX_IDENTITY_MISMATCH")
