# SmolVLA research direction: preserve parallelism without losing reuse

Updated 2026-09-26. Candidate mechanism, not an established paper claim.
Scope: unchanged model, observations, action tokens, denoising steps, and
control protocol. No approximate feature reuse or policy substitution.

2026-09-30 project update: the current working question and execution order are
in the [main-line plan and self-audit](MAINLINE_EXECUTION_PLAN_2026-09-30.md).
Matched full services and collaboration feasibility now precede micro-probes.
This historical note does not define the current project priority.

2026-09-29 followup: the actual action cross boundary now has complete native
head-parallel and numerical ablations. Old output-weight reindexing loses to
the head control and changes rounding; compact groups plus CPU native bit
restoration have a small median increment over that control, with tested bit
parity but unstable tails. See the [latest followup](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md).
This older vision direction is not the current claimed contribution.

2026-09-28 status: this is a historical direction note. Fair dual-view controls
reduce the vision gain to about 4%-6%; direct cross-view stage overlap failed.
Native ten-step residency is largely matched by ordinary native pass-through;
actual action mask/GQA leaves only a small cross-attention clue, not a proven
heterogeneous win. Current decisions are in the
[next mechanism plan](NEXT_MECHANISM_PLAN_2026-09-28.md) and
[conditioning validation](SMOLVLA_CONDITIONING_VALIDATION.md); older context is in the
[reassessment](IDEA_REASSESSMENT_2026-09-28.md),
[flow audit](SMOLVLA_FLOW_INVESTIGATION.md), and
[action attention audit](SMOLVLA_ACTION_ATTENTION_INVESTIGATION.md).

## What the evidence establishes

The core masks in `probe_smolvla_camera_cores.py` select NPU cores, not CPU
threads. The driver counters come from `/sys/kernel/debug/rknpu/load`.
Busy counters are not MAC utilization or measured DRAM bandwidth.

- The current vision graph profile attributes 58.23% of operator time to
  12 `exSDPAttention` calls, with reported core workload 100/0/0. Adjacent QKV
  projections report work distributed across all three NPU cores.
- The denoising graph attributes 42.28% of operator time to attention.
- Full two-view vision execution takes 1740.2 ms with sequential three-core
  calls versus 1360.3 ms with concurrent single-core calls on two distinct
  cores. Both complete the same two inputs; tested outputs are identical.
  Inputs are an export fixture and its horizontal flip, not a LIBERO episode.
- Patch embedding is lowered into an intermediate `[1,768,125,125]` followed
  by selection of `[1,768,32,32]`. This warrants checking execution inflation;
  it does not yet prove which operations a revised graph will avoid.

Profiles are debugging runs, not production latency estimates. Fixed-zero
inputs were used for C operator profiles, and input submission was outside
their measured run loop. Do not compare them directly to Python end-to-end
measurements. Raw profiles and camera results are in `results/smolvla_libero/`.

These observations show that whole-graph execution is not an adequate model
of how work uses NPU resources. They do not establish that dynamic scheduling
is necessary, or that the entire graph reserves every core exclusively.

## Candidate central mechanism

**Share work where branches share weights; separate execution where their
reductions are independent, without paying for an intervening layout roundtrip.**

For the vision encoder, multiple camera views use the same linear weights but
their attention is independent. A possible execution alternates between:

1. Larger shared-weight linear operations over rows from both views.
2. Separate attention over each view or independent head groups, mapped to
   the hardware resources that actually accelerate that operation.
3. A compatible buffer layout connecting these forms without a host roundtrip.

This is exact batching and partitioning, not fewer observations. Camera tokens
must not attend to another camera inside the independent vision encoders.
Conversely, the multimodal backbone intentionally mixes views; the same
camera-isolation rewrite is NOT valid there. Attention heads remain a separate
potential partition dimension, subject to output-projection dependencies.

The question is whether sharing weights and exposing independent reductions
require different execution forms on this NPU. That is a hypothesis to test,
not a conclusion from the two-view experiment. Ordinary batch=2 may already
provide both benefits and must be tested first.

## Why old mechanisms are strong baselines, not disqualifiers

[CoDL](https://www.microsoft.com/en-us/research/publication/codl-efficient-cpu-gpu-co-execution-for-deep-learning-inference-on-mobile-devices-2/)
already studies fine-grained CPU/GPU coexecution and partitioning.
[HeteroInfer](https://arxiv.org/html/2501.14794v2) studies mobile GPU/NPU tensor
partitioning, shape sensitivity, and synchronization.

Neither fixed schedules nor old primitives automatically make a result small.
The required contribution is an observed limitation that survives competent
use of these primitives, and a mechanism that explains and removes that
limitation. Merely adding online scheduling to a stable workload is not enough.
Previously found exact KV projection caching and layout hoisting belong in
the optimized baseline, not at the center of the paper.

## How cloud cooperation could belong

Do not send individual attention operations across a WAN merely because they
are locally slow. Boundary tensors, RTT, and repeated visits can dominate.
Local fine-grained parallelism and remote offload need different granularities.

Cloud participation should be evaluated jointly with the optimized local
execution, not against the old 3.236 s implementation alone. Two legitimate
objectives are:

- Single-request latency on measured links, with full cloud included.
- Cloud GPU service cost or fleet capacity under a declared latency constraint
  and fixed GPU budget. This requires an explicit new workload and queue/load
  measurements; it is not a justification to assume slow cloud service.

A useful cloud mechanism must identify a complete region whose saved local
critical-path time exceeds all input, output, scheduling and conversion costs.
For recurrent regions, count every boundary crossing and every reuse, not just
one invocation. The cloud remains allowed to execute the entire request when
that wins. No deployment mode needs to be made to win artificially.

In particular, do NOT promote cloud-prefix/edge-expert as the default route:
the present local expert takes about 1.086 s while full cloud computation is
about 0.238 s. Shipping approximately 6.1 MB of FP16 prefix KV adds further
cost. Layer streaming can overlap the first expert iteration, not all ten.
Under identical image uplink and an unqueued cloud, the existing figures make
that split unattractive. A server-budget or queue benefit must be measured.

## Three decision experiments before writing a full paper story

1. Compare serial three-core vision, independent-view concurrency, and a
   correctly compiled batch=2 graph. Check all outputs, same precision,
   frequency, input work and memory accounting. If batching solves the issue,
   report it as an engineering optimization rather than inventing a runtime.
2. Compare shared versus separated execution for linear and attention regions.
   Look for opposite winners. Then include every layout, submission and
   synchronization cost in a complete vision-block measurement. If a fixed
   schedule is sufficient, use it; do not force runtime adaptation.
3. Only after the local result survives those controls, measure full-cloud,
   full-local and legal region-offload paths using actual payload sizes and
   network traces. Test a declared fixed-server-budget objective separately
   from single-request latency. Keep any dominated cloud split out of the
   proposed mechanism.

## Provisional paper logic and audit

- Background: exact multimodal inference on asymmetric mobile accelerators,
  with an optional remote GPU under measured resource constraints.
- Limitation: whole-graph execution can hide independent work and assign one
  resource policy to phases with different scaling behavior. Evidence exists
  for our deployment, not for every framework.
- Key idea: choose an execution representation that preserves both shared
  weights and independent reductions, then place complete profitable regions.
- Challenge A: exposing parallelism can sacrifice fusion and locality.
- Challenge B: local gains can be erased by boundary/remote costs.
- Method A: exact merge/separate forms with compatible buffer layouts.
- Method B: whole-critical-path accounting across legal placement boundaries.
- Potential contributions: hardware/graph characterization; an exact execution
  mechanism; end-to-end validation against optimized local and full-cloud
  baselines. No contribution is yet claimed as achieved.

Logical connections are coherent but empirical support is incomplete. The
key unresolved gates are whether ordinary batching already suffices and
whether any nontrivial cloud cooperation is nondominated. Do not draft a
victory-shaped Introduction before resolving these two questions.
