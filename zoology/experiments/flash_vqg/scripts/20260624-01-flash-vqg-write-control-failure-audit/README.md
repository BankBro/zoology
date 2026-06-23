# Flash-VQG write-control failure audit

Plan:

```text
docs/plans/20260624-01-flash-vqg-write-control-failure-audit-plan.md
```

This audit reads existing `cb64-r16` write-control history and manifest files. It does not launch training.

Run:

```bash
python zoology/experiments/flash_vqg/scripts/20260624-01-flash-vqg-write-control-failure-audit/collect_write_control_audit.py
python zoology/experiments/flash_vqg/scripts/20260624-01-flash-vqg-write-control-failure-audit/collect_write_control_audit.py --check
```

Outputs:

```text
docs/artifacts/20260624-01-flash-vqg-write-control-failure-audit/
```
