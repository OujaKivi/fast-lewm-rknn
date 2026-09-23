# Research spine: quality-constrained edge world-model planning

## Claim to test

On fixed-shape, heterogeneous edge accelerators, a planner can meet a latency
budget with less computation than a fixed CEM schedule while preserving task
success. The system should choose computation based on measured marginal value,
not on a hand-written iteration number. This is a hypothesis, not a result.

| Priority | Question | Minimum evidence | Decision |
|---|---|---|---|
| 1 | What actually limits the aligned planner? | Operator and end-to-end profiles at candidate counts 64/150/300, with CPU/NPU timing and numerical parity. | Select one optimization target; do not write hardware-specific kernels without evidence. |
| 2 | Is a quality-aware runtime better than established search baselines? | Paired PushT evaluation against fixed CEM and iCEM at matched latency or candidate budget; report success, P50/P95 latency and energy. | Continue only if it improves the quality-latency frontier. |
| 3 | Does the mechanism generalize? | Reproduce on at least one more official Fast-LeWM task and a second execution setting if available. | Promote from deployment case study to systems claim only if the mechanism transfers. |

## First gate: action-encoder operator profile

The first board run used the released checkpoint, four pinned Cortex-A76 cores,
four PyTorch workers, and the same action encoder as the deployed planner.
After 10 warmups and 50 timed runs, median latency was 18.46/36.81/73.39 ms
for 64/150/300 candidates. At batch 300, `aten::addmm` plus `aten::mm`
accounted for 49.43 ms of self CPU time in a profiled pass, versus 73.39 ms
median wall time in unprofiled passes. This is an approximate comparison, not
an additive decomposition of wall time. The dominant cost is matrix work, not
an obvious Python launch overhead. The raw per-operator profile is in
`results/action_encoder_operator_profile.json`.

Next, measure the same workload under alternative thread counts and identify
which linear layers dominate. Only then test one general optimization (for
example, layout-aware fused projections or a better GEMM backend). A76-specific
packing is optional evidence for portability, not the paper's central method.

## Stop rule

If a standard iCEM baseline matches or dominates the proposed runtime at the
same task-success and latency budget, do not claim novelty from declining batch
sizes. If action-encoder optimization does not improve complete-replan latency,
do not pursue a stand-alone kernel paper on this model.
