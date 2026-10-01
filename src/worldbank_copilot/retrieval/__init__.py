"""Databricks-native document retrieval (Phase 8).

chunking/corpus -> silver.document_chunks; embeddings -> silver.chunk_embeddings;
vector_search -> Databricks Vector Search; lexical (BM25 + RRF); rerank; query
(deterministic processing); retriever (scope, guardrails, citation-ready evidence);
evaluation (metrics, staged experiments); pipeline (notebook 07 orchestration).
"""
