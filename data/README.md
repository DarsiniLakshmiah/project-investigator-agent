# Source data (not committed)

Everything in `data/` except this file is git-ignored. Datasets and PDFs are
never committed; on Databricks the same files live in a Unity Catalog volume
(`WBC_DATABRICKS_VOLUME_ROOT`).

## Current layout (as delivered)

```
data/
  all.xlsx                                   World Bank Projects & Operations workbook
  ibrd_statement_of_loans_and_guarantees_latest_available_snapshot_<MM-DD-YYYY>.csv
  contract_awards_in_investment_project_financing_india_projects_<MM-DD-YYYY>.csv
  P130544/*.pdf
  P179039/*.pdf
  P506272/*.pdf
```

The layout is configured in `configs/environments/base.yaml`
(`data.structured_subdir`, `data.documents_subdir`), so the code does not
hard-code these locations. Source files are treated as read-only.
