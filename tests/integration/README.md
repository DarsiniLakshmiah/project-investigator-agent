# Integration tests

Multi-module tests go here. Mark them `@pytest.mark.integration`.
Tests needing a live Databricks workspace must also be marked
`@pytest.mark.databricks`; they are deselected by default
(`pytest -m databricks` runs them explicitly).
