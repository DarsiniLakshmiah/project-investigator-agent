"""PDF parsing, document metadata validation and parsed representation (Phase 4).

Modules: models (parsed-document schema), parser (``DocumentParser`` interface),
docling_parser (the only Docling-aware module), assembly (blocks/sections/pages),
cleaning (conservative), metadata (type handlers), reconcile (manifest vs document),
document_builder, pipeline (cache/selection/failures), report (parsing quality).

Planned (Phase 5): isr_extractor, results_extractor, risk_extractor, event_extractor.
"""
