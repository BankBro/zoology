# Flash-VQG write-control failure audit artifact

This directory contains a read-only audit of existing cb64-r16 write-control runs.

- `write_control_final_summary.csv`: one row per run.
- `write_control_setting_summary.csv`: grouped setting-level spread and state summary.
- `write_control_step_curves.csv`: validation-step metric curves.
- `failure_taxonomy.csv`: setting-level failure labels.
- `missing_metrics.csv`: metrics absent from historical histories.
- `source_manifest.csv`: source paths and sha256 hashes.

Important limitation: this artifact can only analyze scalars that old runs wrote into `history.csv`. Missing read-side and update-norm telemetry cannot be reconstructed after the fact; see `missing_metrics.csv`.
