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

If operator-specific optimization resumes, first measure alternative thread
counts and identify which linear layers dominate. Only then test one general
optimization (for example, layout-aware fused projections or a better GEMM
backend). A76-specific packing is optional evidence for portability, not the
paper's central method.

## Coexecution gate result

Candidate-level CPU/NPU coexecution is feasible, but its incremental benefit
over the existing tiered schedule is modest. On the same 50 PushT rows, the
tiered baseline and hybrid both succeeded on 44 cases; mean replan latency
changed from 1675 to 1566 ms. Two paired outcomes swapped, so this does not
establish success equivalence. Fixed-observation complete replans changed from
2653 to 2257 ms at population 300, and from 1663 to 1500 ms with the tiered
population. Thus heterogeneous execution is a useful system component, not a
stand-alone paper contribution. The subsequent iCEM comparison and adaptive
allocation gate are reported below; further split-ratio tuning is deferred.

Cloud offload and network model splitting are deferred. Entire-plan offload
after local image-to-latent encoding is the only plausible first comparison:
it sends small latents once per replan, whereas splitting CEM iterations or
model layers across the network would introduce repeated synchronization.
The remote path only helps if its compute plus network tail latency improves
the current local quality-latency frontier without harming availability.

## CEM baseline gate result

The iCEM adaptation and tiered CEM were evaluated on the same 200 PushT rows
with the same hybrid CPU/NPU mapping. In their repeat runs, tiered CEM
succeeded on 182/200 cases at 1499 ms mean replan; iCEM with decay 1.25
succeeded on 178/200 at 1006 ms. The success difference is not established
by this paired sample, and equivalence is also unproven. Their earlier runs
were much slower (2033/1404 ms respectively), so all latency comparisons
need controlled board state. A graph-snap iCEM pilot filled every predictor
slot but improved mean latency by only about 1% against the iCEM repeat.
Raw per-request data are in `results/`; the detailed comparison is in
`README.md`.

The quality-aware runtime gate is now closed. Fixed-shape graph occupancy
matters (decay 1.05 evaluated 4700 real candidates through 6600 predictor
slots), but eliminating padding alone yielded only about 1% lower end-to-end
latency. A locked cost-progress rule extended iCEM from 20 to 30 rounds when
rounds 15-20 improved predicted cost by over 15%. On 200 new rows it beat
fixed 25 rounds 180/200 to 173/200, but on another disjoint 400 rows it lost
332/400 to 341/400. Across both validation sets, adaptive success was
512/600 versus 514/600, mean replan 838 versus 874 ms, and P95 1045 versus
937 ms. The mean latency saving does not compensate for the worse tail and
unreliable quality. There is no defensible paper claim from this CEM scheduler.
Do not tune its threshold further on these same rows. Freeze the CEM baseline
and choose a different core mechanism for the research contribution.

## Stop rule

The CEM allocation stop rule has fired: no robust quality-latency-tail gain
over a simple fixed-budget iCEM baseline was demonstrated. If action-encoder
optimization does not improve complete-replan latency, do not pursue a
stand-alone kernel paper on this model.
