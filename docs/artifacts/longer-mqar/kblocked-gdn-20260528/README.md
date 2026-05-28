# K-blocked GDN longer-MQAR evidence

This directory indexes the 2026-05-28 FLA K-blocked GDN longer-MQAR research evidence.

It is intentionally separate from `official-core-20260526/` because the K-blocked kernel is still research-stage. The evaluation rows are useful for comparing true expanded-K GDN behavior, but the kernel is not upstream-ready: `K=1024,V=64,fp32` still exceeds strict `abs_tol=0.005` for `dk/dbeta`, while passing FLA-style tolerance.

Files:

- `kblocked-gdn-longer-mqar-summary.csv`: compact comparison table copied from the main K-blocked artifact.
- `ek16-ev1-longer-mqar-detail.csv`: formal and batch-search eval rows for `K=1024,V=64`.
- `ek8-ev2-longer-mqar-detail.csv`: formal eval rows for `K=512,V=128`.

Primary source artifact:

- `docs/artifacts/20260528-fla-kblocked-gdn-kernel/`
