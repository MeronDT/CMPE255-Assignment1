# Incremental analysis provenance

These scripts preserve the project’s original small-chunk CRISP-DM workflow. They are intentionally verbose and print the tables used during investigation. The streamlined, final reproducible programs are in `src/`.

Run from the repository root after setting the data path:

```bash
export NBA_MVP_DATA_PATH="data/raw/nba_mvp_stats.zip"
export NBA_MVP_CHUNK_OUTPUT_ROOT="results/analysis_chunks"
python scripts/analysis_chunks/audit_chunk1.py
```

## Suggested order

1. `audit_chunk1.py` through `audit_chunk3.py` — source, target, missingness, duplicates, consistency, and `TOT` investigation.
2. `eda_chunk4.py` through `eda_chunk8.py` — participation, candidate profiles, eras, correlations, and readiness signoff.
3. `data_prep_chunk1.py` through `data_prep_chunk3.py` — audited base table, era-aware engineering, features, and chronological design.
4. `pre_modeling_signoff.py` — leakage and validation contract.
5. `modeling_chunk1_baselines.py` through `modeling_chunk5_clustering.py` — baselines, regularization, nonlinear models, hurdle models, and clustering diagnostics.
6. `final_comparison_checkpoint.py` — validation-period model lock.
7. `build_final_report.py` — regenerate the DOCX report from curated figures.

The locked 2019–2022 test is kept separately in `src/locked_test_evaluation.py` to make the boundary between model selection and final evidence unmistakable.
