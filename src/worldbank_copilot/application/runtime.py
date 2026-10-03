"""Explicit Databricks-only wiring using existing governed adapters and Spark."""

from __future__ import annotations

import os
from decimal import Decimal

from worldbank_copilot.application.service import Application
from worldbank_copilot.common import load_project_registry
from worldbank_copilot.ingestion.documents import load_document_manifest
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.investigation.bounded import InvestigatorAdapter
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter
from worldbank_copilot.investigation.policy import InvestigationPolicy
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
from worldbank_copilot.validation import phase10d_models


def build_databricks_application(spark, settings, *, models_enabled=False, tracing_enabled=True):
    if settings.environment.value != "databricks":
        raise ValueError("production wiring is Databricks-only")
    # Preserved historical preflight, before opening any live reader/index/model.
    phase10d_models.prepare(settings.repo_root)
    from worldbank_copilot.common.dependency_health import check_environment

    health = check_environment(
        settings.repo_root,
        [*phase10d_models.prior.REQUIREMENTS, "requirements-phase10d.txt"],
        settings.config_dir,
    )
    if not health.ok:
        raise ValueError("DEPENDENCY_HEALTH_FAILED")
    protocol = phase10d_models.prior.prepare_protocol(settings.repo_root, dependency_ok=True)
    registry = load_project_registry(settings.config_dir)
    allowed = tuple(
        p.strip() for p in os.environ.get("WBC_APP_ALLOWED_PROJECTS", "").split(",") if p.strip()
    )
    if not allowed or set(allowed) - set(registry.project_ids):
        raise ValueError("explicit registered app project allowlist required")
    policy = InvestigationPolicy(
        cost_ceiling=Decimal(os.environ["WBC_APP_COST_CEILING"]),
        pricing_version=os.environ["WBC_APP_PRICING_VERSION"],
    )
    cost = None
    if models_enabled:
        policy = InvestigationPolicy(
            cost_ceiling=Decimal(os.environ["WBC_APP_COST_CEILING"]),
            pricing_version=os.environ["WBC_APP_PRICING_VERSION"],
        )
        cost = Decimal(os.environ["WBC_APP_MAX_COST_PER_MODEL_CALL"])
        if cost <= 0:
            raise ValueError("positive configured pricing bound required")
    if models_enabled:
        accepted_run = os.environ["WBC_APP_ACCEPTED_10D_RUN_ID"]
        import re
        from pathlib import Path

        if not re.fullmatch(r"10d[1-9][0-9]*", accepted_run):
            raise ValueError("invalid accepted capability attempt ID")
        accepted_artifact = phase10d_models.read_completed(
            Path(settings.artifact_volume_path) / "phase10d_models" / (accepted_run + ".json")
        )
        if accepted_artifact.get("content_identity") != phase10d_models.build_lock(
            Path(settings.repo_root)
        ):
            raise ValueError("accepted capability receipt belongs to a different reviewed protocol")
        approved = {entry["endpoint"] for entry in accepted_artifact.get("schedule", ())}
        if accepted_artifact["summary"]["overall_status"] != "PASS" or any(
            os.environ[key] not in approved
            for key in ("WBC_APP_SYNTHESIS_ENDPOINT", "WBC_APP_CRITIC_ENDPOINT")
        ):
            raise ValueError(
                "selected model stages require preserved passing 10D capability receipt"
            )
    settings_retrieval = protocol.settings
    profile = protocol.profile
    routing = load_routing_config(settings.config_dir)
    entities = EntityIndex.build(
        registry, load_document_manifest(settings.config_dir / "document_manifest.yaml"), routing
    )
    store = ChunkStore.from_table(spark, rp.names(settings, settings_retrieval)["chunks"])
    dense = rp.vector_index(settings, settings_retrieval)
    description = dense.describe()
    if (
        (description.get("status") or {}).get("ready") is not True
        or description.get("name") != profile.vector_search_index
        or description.get("endpoint_name") != profile.vector_search_endpoint
    ):
        raise ValueError("accepted retrieval index is not ready or associated")
    retriever = Retriever(
        store,
        settings_retrieval,
        registry.project_ids,
        embedding_provider(settings_retrieval.embeddings),
        dense,
    )
    search = build_document_search(
        profile, retriever, CrossEncoderReranker(settings_retrieval.retrieval.reranker)
    )
    tools = ToolExecutor(TOOL_SPECS)
    documents = DocumentRetrieval(profile, search, tools)
    rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)

    def context(rid):
        return ToolContext(
            SparkTableReader.for_settings(spark, settings), registry, scales, rules, rid
        )

    service = RoutingService(routing, entities, tools, context, documents=search, semantic=None)
    if tracing_enabled:
        import mlflow

        experiment = settings.require("mlflow.experiment")
        mlflow.set_experiment(experiment)
    return Application(
        service,
        documents,
        settings.config_dir,
        allowed,
        policy,
        synthesizer=DatabricksModelAdapter(os.environ["WBC_APP_SYNTHESIS_ENDPOINT"])
        if models_enabled
        else None,
        critic=DatabricksModelAdapter(os.environ["WBC_APP_CRITIC_ENDPOINT"])
        if models_enabled
        else None,
        investigator=InvestigatorAdapter(os.environ["WBC_APP_INVESTIGATOR_ENDPOINT"])
        if models_enabled and os.environ.get("WBC_APP_ENABLE_INVESTIGATOR") == "true"
        else None,
        models_enabled=models_enabled,
        mlflow_enabled=tracing_enabled,
        worst_case_model_cost=cost,
    )
