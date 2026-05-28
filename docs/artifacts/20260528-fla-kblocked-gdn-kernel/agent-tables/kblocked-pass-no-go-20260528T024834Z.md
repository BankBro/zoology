# K-blocked pass/no-go summary

updated_at_utc: 2026-05-28T02:48:34Z

## Controller Rules Applied

- New blocker threshold from `controller/plan.md`: only escalate and wait when the next action would cross the current gate, affect shared resources, change experiment semantics, or risk polluting formal records.
- Latest shared decisions accepted the `ek16-ev1 retry-ieee` tmp training probe as completed research evidence, with cleanup caveat accepted as a post-completion artifact.
- Latest shared decisions also state that Mendel/Anscombe must not repeat the probe, overwrite returned summaries, push/commit, write formal records, or change source without a new gate.

## Verdict

- **Go for tmp research/training-probe evidence**: true large-K path can run one full `ek16-ev1` fp32 4-epoch probe under the authorized tmp-only scope.
- **No-go for upstream-ready correctness**: `K=1024,V=64,fp32` still relies on FLA-style gradient tolerance for `dk/dbeta`; this is not a final upstream correctness claim.
- **No-go for official ledger/formal report**: the probe is explicitly tmp-only and must not be written to official training ledger, weekly slices, or formal docs.
- **No-go for more experiments under current gate**: no second probe, fallback `ek8-ev2`, benchmark, or new training is authorized.
- **Ready for controller review of patch/merge strategy**: source changes and small manifests are organized enough for a future explicit code-review or patch-export gate.

## Evidence Chain

- True Triton Phase A source: new large-K hidden-state fwd/bwd path in FLA, public K>256 autograd fallback removed/disabled.
- CUDA correctness Phase A: 11/11 fp16 cases passed for `K=512,V=128`, `K=1024,V=64`, and K<=256 regression.
- Benchmark Phase A: large-K kernel-level timing collected before the training gate; not a training-quality claim.
- Zoology wrapper smoke: `GatedDeltaNetExpandedK` forward/backward passed for `H=2,K=512,V=128` and `H=2,K=1024,V=64`, true large-K path hit.
- Shared-memory repair: `num_stages=2` removed fp32 `K=1024,V=64` Triton shared-memory error.
- FP32 numerical diagnosis: `input_precision="ieee"` reduced TF32-driven mismatch; FLA-style tolerance passed for required cases, while strict `abs_tol=0.005` remained slightly exceeded for `dk/dbeta`.
- Retry training probe: `ek16-ev1`, `H=2,K=1024,V=64`, seed=123, data_seed=123, b64_ga4, fp32, 4 epochs, no early stop, `GDN_KERNEL_DTYPE=float32` completed.

## Training Probe Result

- Run id: `gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep-retry-ieee-20260528T020346Z`.
- Launch id: `flash-vqg-20260528-kblocked-probe-ek16-ev1-retry-ieee-20260528T020346Z-2026-05-28-02-06-08`.
- Manifest status: `completed`, error null.
- Wall clock: `741.254436` sec.
- Final validation:
  - `valid/accuracy=0.974`.
  - `valid/loss=0.251`.
  - `valid/mqar_case/accuracy-1024x256=0.792`.
  - `valid/mqar_case/accuracy-512x128=0.998`.
  - `valid/mqar_case/accuracy-512x64=0.999`.
- Health: no OOM, CUDA error, NaN/Inf, dtype fallback, shared-memory error, or training failure before completion.
- Caveat: after manifest completion, wrapper Python remained alive and held GPU memory at 0 percent utilization; Ctrl-C cleanup released GPU and produced a post-completion `KeyboardInterrupt` traceback. Controller accepted this as cleanup artifact, not training failure.

## Current Resource State

- 3090 GPU after cleanup: 28 MiB used, util 0%, P8, no compute process.
- No tmux session remains for this probe.

## Current Dirty Scope

- 3090 FLA worktree, branch `codex/fla-kblocked-gdn-kernel`, commit base `19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5`:
  - `M fla/ops/common/chunk_delta_h.py`.
  - `M fla/ops/gated_delta_rule/chunk.py`.
  - Diff stat: `2 files changed, 327 insertions(+), 2 deletions(-)`.
- 3090 zoology worktree, base `dd522a77f8b101e90e7204b1b381c28c99cf0bbd`:
  - `M zoology/mixers/gated_delta_net.py`.
  - `?? artifacts/`.
  - Source diff stat: `1 file changed, 3 deletions(-)`.
- 3090 Flash-VQG worktree, base `603d2d603fb5389b5eaaad1ccfbe569c5fc023b4`: clean.

## Recommended Next Gate

Request a separate `kblocked_patch_review_and_export` gate before any source integration. Suggested allowed actions for that gate:

- Generate patch files from the 3090 FLA and zoology worktrees.
- Exclude checkpoints, raw logs, swanlog, caches, profiler outputs, and large artifacts from source commits.
- Include only small tmp manifests/summaries in this agent directory.
- Review whether FLA source should be split into separate commits: large-K hidden-state path, shared-memory/IEEE numerical repair, and public dispatch cleanup.
- Review whether zoology wrapper `head_first=False` removal should be a separate compatibility commit.
- Decide whether to push to BankBro fork, transfer patches to 2080ti, or keep as research branch only.
