# SmolVLA systems idea candidates

Updated: 2026-09-30. This is a decision log, not a list of claimed contributions.
Latest project plan: [one main line, experiment priorities and self-audit](MAINLINE_EXECUTION_PLAN_2026-09-30.md).
Earlier executed subplan: [visual services and boundaries](PLAN_VISION_BOUNDARIES_2026-09-30.md).
Latest experiment: [actual vision service and stage-geometry counterfactuals](SMOLVLA_VISION_SERVICE_VALIDATION_2026-09-30.md).
Earlier full-policy result: [recorded replay, tail trace and lossless payload](SMOLVLA_POLICY_REPLAY_VALIDATION.md).
Earlier boundary followup: [numerical/head/bit-copy audit](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md).
Earlier boundary experiment: [actual cross-Attention boundary validation](SMOLVLA_CROSS_BOUNDARY_VALIDATION.md).
Earlier full-flow baseline: [actual conditioning validation](SMOLVLA_CONDITIONING_VALIDATION.md).
Earlier reading audit: [discussion and optimization gates](VLA_READING_ABSORPTION_2026-09-28.md).
Earlier audit: [five-route reassessment](IDEA_REASSESSMENT_2026-09-28.md).
Scope: SmolVLA as the primary VLA; preserve the model, input observations,
action count, denoising steps, and execution semantics. VLM/LLM mechanisms are
welcome when the hardware and dependency argument survives this scope.

Status key: `test` = a decisive experiment is running or next; `open` =
plausible but missing a necessary measurement; `hypothesis` = a distinct,
untested mechanism with a named experiment, not automatically a baseline;
`baseline` = useful known-mechanism engineering control;
`reject` = a specific claim contradicted in its tested scope, or out of scope.
No experiment is currently running. The ranks below are historical, not the
current execution order.

## Current project decision: one main line, not a queue of micro-probes

Working question: **Can local NPU and remote GPU collaboration complete an
unchanged multi-view flow VLA action chunk sooner than optimized all-local and
streaming all-cloud execution on a constrained link?** This is a falsifiable
research route, not an established contribution. The proposed coupling between
placement, payload and local execution must earn an increment over ordinary
offloading and the best static plan; a best-local-configuration reversal is not
an observed fact or a mandatory invented motivation.

| Current priority | Required evidence / deliverable | Decision |
|---|---|---|
| P0: matched complete services and dependencies | GPU vision/prefill/whole flow, all-local and streaming all-cloud with one request boundary; reuse completed local matrix | Next main task; no old CUDA closed-loop number as the new remote service |
| P1: complete collaboration feasibility | Legal branch/join plans, predefined network range, real payload/codec/link costs, optimistic bounds then actual link validation | Before scheduler construction; stop this latency route if even the optimistic hybrid is dominated |
| P2: one critical-path mechanism | Joint vs placement-only/local-only/best-static controls; identical-work overlap ablation | Conditional on P1; static offloading absorbing the gain is not a new adaptive mechanism |
| P3: integrated system evidence | More episodes/instructions, quality, network traces, tails, thermals and isolated memory/energy | Conditional on a legal, important mechanism; no present claim of completion |

The historical C04 record (same-observation branch collaboration) supplies the
initial feasibility question, not a ready-made paper solution. Stage geometry
is now a bounded numerical support item, not project priority one. C17 (native
bit bridges) is retained as local mechanism evidence, with no all-layer expansion.
C08/C13 (within-attention backend diagnostics), C15 (prefix/first-step overlap),
C05/C10 (hedge/migration) and C16 (strict speculation compatibility) are paused
unless the full trace makes them necessary. Historical IDs index evidence;
they do not denote separate promised modules or automatic fallback stories.
This documentation update contains no new hardware results.

## Evidence and earlier candidate decisions

Full-policy importance gate (2026-09-29): P0 replays three actual LIBERO demonstration frames,
with both cameras, 32 action layers and ten dependent original FP32 Euler steps.
Two independent sessions, eight rotating trials per frame/plan: fused whole-camera
concurrency plus resident action is about 2135-2138 ms; C06 reaches 2078-2084 ms,
2.43%-2.78% lower whole-policy medians. Tested dependent velocities/latents match bits.
Vision remains 52.68%-53.11%, action 33.84%-34.23%. This is offline replay, not task success.
C17 replays observation-driven K/V and suffix trajectories, with CPU FP32 reconstructed
layer-0 hidden: 30 tested steps pass bits and local ten-consumption latency still falls
about 10.2%, but saves only about 1.2 ms/block. Do not enlarge to all 32 layers on that
evidence. Three 500-trial tail diagnostics implicate the synchronous NPU call path,
not native bit-copy or worker startup; the kernel/driver cause remains unknown.
Actual lossless PNG is 57-66 KiB/view, versus about 111 KiB for lossless native connector
storage after compression. C04 must earn its advantage through dependency/resource/link
overlap, not a default "features are smaller" premise. Its local parallel-contract
hypothesis remains open after the actual service update below.

Current decision supersedes the earlier ranks below: **C14 completed and demoted
to a baseline-sized result**, not the next main experiment. Ordinary expanded
consumer-ready caching passes the full flow gate (747.77/749.43 -> 692.65/693.32 ms,
two 12-repeat sessions, all tested step velocities/latents bitwise identical).
Compact repeat/grouped consumers lose in the old real layer-1 fragment. C17 now
has a different, tested producer-to-consumer representation mechanism, but its
initial 11%-12% gain is beaten by ordinary native head parallelism. Weight
reindexing changes output-projection rounding (260/512 stress cases bit-equal).
Restoring original output order fixes tested bits but is slower. Native
bit-copy output restoration first adds ~5%; bidirectional bridges then retain
original Q/RoPE and O weights, reaching 10.27-10.29 ms versus level-3 expanded
head parallelism's 11.46-11.50 ms, a 10.3%-10.7% median increment. The final
independent 512-case check passes bit parity. p95 is worse in one session. Full-policy
importance, generality, tail causes and novelty remain open. This is neither
a deployed full-policy replacement nor an established paper line.
C04 previously remained
a separate lossless payload/latency-region audit only; its feasibility question
is now organized by the main-line plan above. C15 previously got a dependency and
ceiling check, not a full streaming implementation. C16 is an AR/VLM reserve,
not directly compatible with the current flow sampler. C13 is now only an
auxiliary backend probe, not the next main experiment. The native-loop strong
baseline is complete; C09 is not a paper mechanism. Ordinary C06 projection
hoisting remains a baseline, not newly claimed by C14.
C01 remains a measured vision mechanism and integration control; C12 is
optional engineering work, no longer the next main experiment.

**2026-09-30 update:** actual three-frame/two-camera service matrix completed,
two independent sessions, eight rotating trials per plan/mode. Including native
input/output, single-camera head improvement is 32.4%-32.8%; dual-camera head
serial versus fused camera concurrency is only 6.17%-6.35%. Head/full twelve-layer
and swap/restore gates pass, but split/fused connector still differs. Head wins in
both branch counts: the earlier proposed best-configuration crossing is **not proven**.

The actual last-layer consumer does not show a large restore matrix. Equivalent
whole-consumer 16x64/32x32/64x16 native geometry is bit-equal but slower. Cutting
after normalization preserves B=1 2-D MLP geometry and lowers the complete
prefix+MLP segment from about 18.9 to 15.8-16.0 ms, extra call charged. All tested
spatial MLPs match the flat split anchor, but not the original consumer. ConvAdd
becomes Conv + Add; rounding provenance is not closed. A stronger simple control,
reshape only after Norm inside one graph, is also slower: the compiler moves it
before Norm and converts H into batch. This is a stage-geometry/compiler-boundary
hypothesis, not a lossless speed claim, not a full-policy result or a paper line.
Whole-geometry and simple single-graph reshapes stop here. Within this support
item, preserving original numerics is required before any layer expansion;
it is no longer the next project experiment. GPU/simulator
access is restored; no new cloud service or closed-loop result yet.

**Execution order is now P0 -> P1 -> conditional P2/P3 in the main-line plan.**
The earlier P0 recorded local replay is completed; the new P0 fills the missing
matched remote services and request-level accounting. Do not repeat completed
local service/trace work or promote stage-geometry shape sweeps to the main task.
Native bit-copy passes local median/numerical gates, not integrated policy or
tail/generalization gates. Vision rewrites need their own numerical gate.
There is still no established CCF-A contribution; the project now has one
working question and a bounded go/no-go sequence instead of competing probes.

| Earlier Rank / ID | Candidate / central question | Evidence already in hand | Strongest simple explanation or baseline | Next decisive test and kill condition | Status |
|---|---|---|---|---|---|
| bounded support / stage geometry | **Which compiler boundary can preserve a useful stage-specific geometry without changing arithmetic?** | Same 1024 positions/weights, actual last-layer inputs from six images, 40 rotating trials. Whole 64x16 consumer ~41.9 ms vs original ~18.9, all tested bits pass. Flat normalization prefix + spatial MLP ~15.8-16.0 ms, extra call charged; spatial/flat split bits pass, original-consumer bits fail (max 0.5). In-graph post-Norm reshape loses (~42.4-42.5 ms at 64x16); profile places reshape before Norm and maps H to batch. FC2 improves in the true split geometry; no physical DDR/occupancy claim. | Original/recompiled consumer; rank-only 1x1024; whole-consumer factoring; matched flat split; single-graph local reshape. Ordinary graph partition/layout specialization are old. Splitting itself changes ConvAdd fusion, so speed is not yet a legal replacement. | At most an initial 1-2 workday numerical provenance audit, subordinate to P0/P1; no more shape sweeps or full-layer expansion. Stop if speed needs relaxed numerics. Reopen integration only with a legal interface and full critical-path importance. | demoted to bounded support; no legal original-anchor speedup or paper claim |
| local hold / C17 | **Native bit bridges instead of bending the arithmetic producer.** Can compact Attention geometry coexist with original Q/RoPE and O arithmetic through cheaper native boundaries? | Complete Q/RoPE/Attention/O/residual, 19 controls. Old weight permutation fails bits and loses to head parallelism. Two strict compact combinations lose. Profiles locate CPU reshape/960x960 recovery Conv; C8 lanes permit 120 contiguous bit-copy tiles/direction. Matched level-3 S13-S15: 10.27-10.29 ms versus head control 11.46-11.50, all prep/sync/dispatch charged; independent 512-case bit audit passes. Original Q/O unchanged. State 583680->194560 bytes. Observation-driven K/V and suffix replay with CPU-reconstructed hidden: 30 steps pass bits, 10.52-10.53 versus 11.72-11.74 ms, 60 trials/input. Only ~1.2 ms/block, not integrated C17 policy. Three 500-trial diagnostics narrow tails to NPU calls, not bit copies/wake; cause unknown. | C06 ordinary ready; native 5+5+5 head parallelism with persistent workers; compact 2+2+1 group parallelism with compiler restore; original-head-order packing; one-direction restoration; bidirectional bridges without group parallelism; static-only and matched level-3 controls. GQA, head parallelism, CPU transposes/native interop alone are not novel. | Do not expand all 32 layers on the small measured increment. Find another important same-cause boundary/shape or prove integrated gain over strongest whole graph. Preserve FP32 Euler and dynamic self masks/KV. Do not infer DDR/capacity from state size; no root-cause or stable tail guarantee yet. | local mechanism survives observed conditions; importance gate limits expansion; integrated policy/generality/novelty open |
| 1 / C01 | **Fusion versus usable parallelism.** Can independent camera/head work become parallel without paying a host-layout boundary at every attention? | Historical B1 head gain 31.4%, dual-view increment 4%-6%. Now actual six images/two sessions, native I/O included: fused single 766.43-768.27 ms vs head 516.03-518.65; fused camera concurrency 1102.94-1105.20 vs head camera-serial 1033.63-1035.12. Actual head single gain 32.4%-32.8%, dual 6.17%-6.35%. Extra camera concurrency <1%. All head/full twelve-layer/state gates pass; split/fused connector MAE 0.01376-0.01817, max up to 0.3125. Full-policy fused vision ~53%; head-policy quality untested. | Orca/CoDL/HeteroInfer/Rammer already cross inter-/intra-operator boundaries. Head parallelism/native I/O alone are not new. Fused whole-camera concurrency absorbs most single-view win; direct stage complementarity unsupported. | Service matrix completed. Keep fused policy control; audit complete head-policy numerical/quality gate independently. Stage-geometry candidate is a different boundary question, not more head threads. No multi-view scheduler before a measured remaining mechanism increment. | actual-input service/state validation completed; head policy quality/generality open; scheduler paused |
| 2 / C12 | **Different parallel contracts inside the MLP.** Can output-channel partitions of FC1+pointwise activation feed full FC2 through a native activation buffer? | Original 12-layer profile: FC1+activation 87.23 ms, single-core without mask7 scaling; FC2 266.06 ms on one core versus 108.76 ms on three. No partitioned FC1 timing yet. The optimistic threefold FC1 ceiling saves ~58 ms/view before costs, using the old profile rather than a measured current-pipeline bound. | Ordinary tensor partitioning, Megatron-style column-parallel first GEMM/activation, compiler-native multi-core GEMM, and repaired fused consumer. Not VLA-specific, and single-core mapping is not proof of a defect. | Optional bounded whole-consumer test only. Reject as a paper centerpiece without a broader demonstrated mechanism; do not divert the next main experiments to this remaining hotspot. | optional engineering; downgraded, not next main test |
| - / C02 | **Useful-work-aware graph lowering.** Which equivalent representation avoids compiled work/constant inflation? | Actual B2 zero-mask lowering isolated: removing 12 verified-zero mask additions changes weight-region bytes 802625856 -> 198646080; delta is 12 times 50331648 bytes, exactly a B2/12-head/1024-square FP16 score-shaped allocation per layer. Standard optimization level 3 retains the inflated region in the actual attention probe. Full B2 latency 1739.01 -> 1722.39 ms (~1%); tested full-graph outputs identical. Exact patch rewrite remains a separate ~60 ms/view improvement. | Constant elimination and graph canonicalization are ordinary compiler controls. The memory anomaly is real but is NOT evidence of a large latency opportunity or parameter replication. | Keep zero-mask repair in the strong baseline; do not present it as the speed main line. Investigate broader lowering only if another structural case appears. | baseline; causal path located |
| - / C03 | **Within-operator backend reversal.** Does attention, unlike large GEMM, run better as a CPU/NPU cooperative tile or head group on RK3588? | Four A76 CPU threads: standalone FP32 SDPA 40.98 ms; torch FP16 SDPA 413.13 ms. Native NPU full attention ~41.55 ms, parallel 6+6 heads ~24.24 ms, before full-block integration. Whole-CPU placement has no measured advantage once transitions are charged. | Optimized CPU kernels, native NPU head concurrency, or a better fused NPU kernel. Torch FP16 performance is not a hardware impossibility result. | Deprioritize whole-CPU attention. A CPU/NPU tile experiment requires a separately demonstrated normalization advantage and complete transition accounting. | negative for naive CPU placement; C08 still unproven |
| independent feasibility / C04 | **Same-observation branch collaboration and local parallel contracts.** Does moving one camera also change the best remaining local head/camera execution dimension? | Branches independent until prefill. Strong local policy ~2.08 s, vision ~1.108 s, post-vision ~0.99 s. PNG 58270-67358 bytes/view vs lossless native FP16+zlib 113842-113986; features are NOT a smaller uplink default. Actual head service wins in both single/two branches, so no best-configuration crossing confirmed; quality remains open. GPU host/checkpoint access restored 2026-09-30. Old CUDA 0.238 s is not matching actual cloud service. No network/placement win. | Streaming full cloud with identical lossless source codec and upload/compute overlap; optimized local with numerical gate; best static branch/core configuration; ordinary hedge. Hierarchical parallelism/offloading are old; combination alone is not novel. | Actual local service matrix is done. Measure uplink/downlink, cloud branch/full service/queue and join. Require a region beating strong full local and streaming full cloud, plus an increment over best static. Test local-contract coupling rather than assume it. Stop otherwise; no migration runtime first. | payload motivation corrected; contract switch unproven; independent service/region audit, not fallback |
| 9 / C05 | **Exact hedged placement.** Can local NPU progress and a cloud request overlap on the same observation, then complete the first valid exact path without paying most redundant work? | Local compute is currently much slower than cloud compute; no evidence yet of a useful latency crossing. | Full cloud; optimized local; simple parallel duplicate inference. | First establish a realistic network/queue distribution where local can win; then measure cancellation and graph preemption cost. Reject if it only helps contrived delay traces. | open, low priority |
| - / C06 | **Request-scoped consumer-ready conditioning.** Cache expert-side K/V projections and native-layout inputs for all ten denoising steps. | Synthetic full flow: two 12-repeat sessions resident 747.77/749.43 -> cache 692.65/693.32 ms, ~10.45 ms preparation included, tested velocity/FP32 latent bits pass. Now actual recorded two-camera policy, two eight-repeat sessions/frame: resident ~2135-2138 -> C06 ~2078-2084 ms, additional 2.43%-2.78% whole-policy median reduction. All tested dependent velocity/latent bits and observation restore pass; ~5 ms raw prefix packing/~11 ms NPU prep charged. | Ordinary invariant hoisting/consumer-ready cache; no new algorithm/training/reduced target work. Seven/eight contexts coexist, not isolated peak RSS. Three recorded frames, not task success. | Keep as strong policy baseline. More tasks, closed-loop and isolated memory/energy remain validation, not automatic paper contribution. | baseline; full flow and recorded full-policy integration verified |
| - / C07 | **Adaptive action count, skipped observations, approximate visual reuse.** | Can improve latency but changes the amount or quality of model work and confounds this systems question. | Strong existing adaptive-chunk and reuse papers. | No experiment under the present scope. | reject |
| 3 / C08 | **Matrix/vector streaming instead of backend-wide attention placement.** Can dense QK/PV tiles use NPU matrix engines while CPU handles exact normalization, without materializing the full score matrix? | Attention is the largest measured visual hotspot. No evidence yet that CPU normalization is faster, or that the current fused kernel's limitation is vector work. | Existing FlashAttention online softmax, CoDL operator chains, HeteroInfer and a better NPU fused kernel. | Benchmark QK, stable softmax, PV and native buffer transitions separately; then pipeline exact head/tile work across CPU/NPU. Reject if synchronization/shared-DRAM contention erases the bound. Never infer vector bottlenecks from NPU load alone. | open; exploratory |
| 5 / C09 | **Keep an observation epoch resident across the flow loop.** Can conditioning stay in native memory while the changing latent advances, without unrolling and duplicating compiled constants? | Six modes, 12 rotating trials on the unchanged ten-step graph: prepacked FP32 ordinary input 1029.21 ms; persistent native/default sync 793.24 ms; ordinary prepacked native pass-through 794.65 ms. Input-stage work falls further with binding but total gain is not stable. A separate strict uint32 storage check passes all step velocities/latents and meaningful prefix switch/restore. | Ordinary native layout preparation plus pass-through absorbs the apparent large residency gain. No static projection inside the graph was removed. Dirty-sync variant shows no benefit. | Retain native/pass-through strong baselines; stop treating persistent binding as a main contribution or tuning flush flags. Cold/warm graph projection is a separate C06 question. Keep FP32 latent/Euler unchanged; no lossy unroll. | baseline; residency speed claim rejected against strong control |
| 6 / C10 | **Small committed-state migration with conditioning already resident.** Move exact latent/step metadata rather than restart flow. | True step boundaries exist; dual-end conditioning setup costs are not paid for. | SpotServe progress-preserving migration, whole-request hedge and restart. | Count all conditioning duplication, transfer and extra compute; require a real end-to-end crossing. Reject an evaluation that assumes free dual-end KV residency. Cross-backend bitwise equality is not guaranteed. | open; low priority |
| - / C11 | **Graph-wide core reservation prevents other work.** Release apparently idle cores by phase leases. | Standalone full SDPA plus core-1 companion: mask7 makespan 79.0 ms vs mask1 79.5 ms; companion completes in 42.6/43.5 ms. It progresses under mask7. | SDK scheduling already admits work on other cores. | The broad reservation claim is not supported and must not motivate a paper. Whole-vision graph behavior could be tested separately, but cannot override this counterexample. | reject in tested scope |
| auxiliary / C13 | **Short queries over persistent conditioning.** Does action attention warrant a different backend/data boundary than long vision attention? | Actual checkpoint CPU activations and mask/GQA, two suffixes: 16 self layers Q50/K199 have dynamic K/V and causal suffix mask; 16 cross Q50/K149 have invariant K/V. FP32 fused including dynamic GQA/layout takes 1.162/0.853 ms; preprepared cross 0.546 ms. Old NPU profile ~1.080/0.849 ms is still NOT a paired comparison. Whole 32-layer CPU velocity differs by up to 1.91e-6, not bitwise equal; no device boundaries or rollout. | Native all-NPU flow/pass-through, split-but-all-NPU boundary control, native head partition, same-protocol optimized CPU/GQA kernel, HeteroInfer-style shape placement. Caching and ordinary fused SDPA are baselines, not novel. | Only when a broader dependency/data mechanism needs a backend diagnostic: one real cross producer/attention/consumer block, all boundaries charged. The ~0.3 ms/layer clue alone is small and mostly generic; no longer the next main experiment. | open auxiliary; downgraded after VLA-relevance review |
| completed / C14 | **Which conditioning representation should persist?** Does compact conditioning outperform ordinary consumer-ready caching? | Real layer-1 fragment, 40 rotated repeats: original ten-consumption 17.138 ms; compact repeat 19.516; expanded token-major 19.182; compact grouped 28.339; expanded consumer-ready 13.775. All tested NPU fragment outputs match fused bits. Compiled projection reports 320->960, duplicate head outputs match compact repetition, weight region 1231104 versus compact 411904 bytes. Grouped attention core 862->541 us but query reorder 1206 us. Full C06 cache wins 7.37%-7.49% versus best original control. | Ordinary exact cache plus consumer-ready layout absorbs the useful win. No demonstrated capacity bottleneck from 3x state; physical MAC/DDR still unmeasured. Do not rename GQA/caching as new or use faster inner kernel to hide slower full path. | Stop advancing this version as a paper main line. Reopen only for a new measured obstacle/different mechanism, not more tuning of these losing compact paths. C04 is independent, not an automatic fallback. | baseline-sized discovery; compact latency claim rejected in tested fragment; not a paper main line |
| ceiling audit / C15 | **Does the whole prefix have to finish before action starts?** Can ready per-layer K/V release first-step expert work? | The deployed wrapper waits for all 64 prefill outputs; model layer dependencies suggest a possible finer frontier, not a demonstrated overlap. Current flow depends on the prior FP32 latent at each step. | Whole-stage pipeline, native prefill outputs/inputs, ordinary dataflow scheduling. Same NPU/DRAM can erase concurrency. | Draw dependencies and use a unified trace. Simple prefill/first-step overlap saves at most min(prefill, first_step), not ten times that. Stop if the ceiling is too small or boundaries erase it; no dozens-of-graphs implementation first. | open, lower priority; ceiling check only |
| compatibility reserve / C16 | **Approximate proposals, exact current-condition target.** Can draft-side condition reuse hide new-observation startup without relaxing target verification? | Literature supports this separation for AR VLM; no compatible verifier measured for the unchanged deterministic SmolVLA flow. Similarity-only target KV reuse and relaxed action acceptance remain excluded. | ParallelVLM parallel prefill/decode; strict AR SD; training-free retrieval proposals; exact stochastic diffusion coupling under its own sampler assumptions. | First establish a strict verifier and checkpoint/runtime compatibility. No automatic AR-to-flow acceptance transfer, added stochasticity, new trained drafter, or model switch. Only then charge conditioning startup, target verification, rollback and wasted work. | open reserve for AR/VLM; not current SmolVLA implementation |

## Decision rules

1. An operator being below the Roofline ridge is not a problem by itself.
   Compare its measured throughput with the attainable compute or bandwidth
   roof for its arithmetic intensity. Driver NPU busy percentage is not MAC
   occupancy, and RKNN `RW(KB)` is not measured external-memory bandwidth.
2. Test simple compiler/runtime controls before naming a new scheduler:
   ordinary batching, separate graphs, core masks, conversion options,
   optimized native I/O, and exact loop-invariant hoisting.
3. Require a single mechanism to explain the gains across multiple shapes,
   stages, or workloads. A fixed schedule may be excellent if it follows a
   real hardware law; online adaptation is not automatically novel.
4. For cloud ideas, report both single-request latency and server resource
   cost under explicit conditions. A remote GPU is not a weak baseline merely
   because the proposal focuses on the NPU.
5. Separate known useful controls, data-refuted specific mechanisms and
   untested mechanisms. Missing evidence alone is not evidence of triviality.
   A small implementation or old primitives do not preclude a contribution;
   require an important obstacle, a non-obvious cause, a mechanism increment
   over the strongest reasonable control, and complete attributable evidence.
6. Never construct an end-to-end number by summing medians from different
   sessions. Preparation overlap is bounded by actual preparation cost;
   prefix/first-step overlap cannot save all ten dependent action steps.

## Iteration log

- **2026-09-29, actual policy/importance and payload gates:** Three recorded
  demonstration frames now run the complete unchanged dependent flow. Two
  eight-repeat sessions per frame show C06's additional 2.43%-2.78% whole-policy
  median reduction over resident action, with step velocity/FP32 latent bits
  retained. Vision is still about 53%. Observation-driven C17 boundary replay
  passes 30 step checks and gains about 10.2% locally, only 1.2 ms/block; all-layer
  expansion is paused pending an important additional boundary. Three 500-trial
  diagnostics distinguish stable CPU bit copies/worker wake from slower NPU calls,
  without claiming hardware causality. Actual compressed native features are
  larger than lossless PNG. C04 now needs a branch-count/local-parallelism and
  streaming-cloud winning-region test, not a weak raw-FP32 image baseline.
  See [replay and payload](SMOLVLA_POLICY_REPLAY_VALIDATION.md).

- **2026-09-29, final bidirectional native bridges:** Matched eight level-3
  producer/consumer/attention graphs do not absorb the one-direction ~5%
  increment. Reversing the same C8 tile copy for Q input then removes the need
  to reindex Q weights or propagate its layout through RMSNorm/RoPE. Final
  S13-S15, 19 controls, 40 repeats each: original Q/O + two native bit bridges +
  compact group parallelism 10.281/10.269/10.290 ms versus matched head control
  11.463/11.502/11.500 ms, 10.3%-10.7% lower medians, all sync/prep charged.
  A new independent level-3 512-case audit passes bits, including prefix
  switch/restore. S13 p95 is worse: no tail guarantee or full-policy claim.
  The original hypothesis that the producer must directly emit the chosen
  attention layout is revised, not preserved despite contrary measurements.
  See [final followup](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md#8-最后的反证上游也不必强扭成新表示).

- **2026-09-29, C17 followup and native bit restoration:** CPU stage capture and
  NPU ablations locate weight-permutation rounding at output projection.
  Native 5+5+5 head parallelism beats initial C17, so its fastest-path claim is
  withdrawn. Two strict compact parallel combinations also lose. Profiles
  reveal restoration Conv/CPU reshape; raw C8 storage instead permits 120
  contiguous bit-copy tiles. With all sync/dispatch/preparation charged, the
  compact parallel + bit-copy variant is 5.1%-5.4% faster in median than native
  expanded head parallelism, S7-S9 each 40 repeats. Extended 512-case bit
  audit passes; p95 does not consistently improve. Keep old deployment,
  prioritize actual-policy importance and tail diagnosis, not a large runtime.
  Details: [cross followup](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md).

- **2026-09-29, C17 complete cross-boundary test:** Added the actual Q producer,
  RoPE and output projection rather than extending the losing C14 consumer.
  Nine controls distinguish automatic fusion, query ordering, static layout
  preparation and dynamic representation propagation. Three independent
  sessions yield 11%-12% lower ten-consumption latency versus ordinary ready,
  with preparation and native boundaries charged. The compiler-added output
  recovery Conv disappears, but a CPU reshape remains. Rare last-bit output
  differences prevent a strict lossless claim; keep C06 deployed baseline
  unchanged. Next gates are numerical isolation and complete native
  head-parallel control, not all-32-layer integration. Details:
  [cross-boundary validation](SMOLVLA_CROSS_BOUNDARY_VALIDATION.md).

- **2026-09-28, post-validation exploration update:** Separate C06 known
  engineering gain, C14's refuted compact consumer and new untested C17.
  Prioritize a real-policy trace and one actual Q-producer/attention/output
  block; retain C08 phase diagnostics and C04 payload feasibility as independent
  candidates. Rechecked FlashAttention, Welder, HeteroInfer and RoboECC primary
  sources; no novelty claim established. No new hardware experiment in this
  planning update. Details: [next mechanism plan](NEXT_MECHANISM_PLAN_2026-09-28.md).

- **2026-09-28, actual C14 falsification and C06 integration:** Built five real
  layer-1 controls with native continuation. Compact/grouped versions lose;
  ordinary expanded consumer-ready cache wins. Integrated it into all 32 action
  layers and ten dependent FP32 Euler steps: two sessions give 747.77/749.43 ->
  692.65/693.32 ms against best original native control, preparation charged.
  FP32 ONNX/independent ORT and all tested NPU step velocities/latents pass
  strict bit parity, including meaningful prefix switch/restore. Keep as C06
  strong baseline; stop promoting C14's current compact mechanism into a paper.
  No new closed-loop, network, energy or multi-device claim. See
  [conditioning validation](SMOLVLA_CONDITIONING_VALIDATION.md).

- **2026-09-25:** Verified exact static K/V projection opportunity and CPU/NPU
  preparation overlap. Downgraded it to a baseline-sized optimization after
  considering fused-NPU and native-cache alternatives.
- **2026-09-26:** Added NPU operator profiles and per-core load probes. Measured
  two-view NPU-core concurrency advantage, identified patch-embedding work
  inflation, and prioritized ordinary batch=2 as the nearest falsifier.
- **2026-09-26:** B2 failed to reduce two-view latency; queried compiled memory
  rather than inferring memory from artifact size. Exact patch space-to-depth
  reduced one-view latency by about 60 ms. Patched two-view concurrency reached
  1172 ms, but cross-graph comparisons are not a paired closed-loop experiment.
- **2026-09-28:** Verified standalone exact head concurrency, then falsified the
  broad core-reservation explanation. Rejected the initial NCHW probe because
  the runtime mutated input layout; retained only preconverted native-NHWC runs
  with repeated-full output parity. Standalone zero/derived-zero masks did not
  reproduce full-vision B2 memory inflation. Literature review identified Orca
  as direct prior art for selective batching; opened C08-C10 without claiming
  novelty. Detailed rationale: [taste expansion](IDEA_TASTE_EXPANSION.md).
- **2026-09-28, vision-focused follow-up:** Located actual B2 zero-mask lowering
  as the constant inflation path, but measured only ~1% full-vision latency
  improvement; downgraded C02 to baseline. Changed attention boundaries from
  semantic rank-3 rows to channel-major 4D native interfaces; validated FD-offset
  head views and a complete actual layer-0 block. Three-way heads cut latency
  62.98 -> 42.24 ms against a repaired native fused control. This passes a local
  mechanism gate, not a full-vision or paper-novelty gate. Detailed evidence and
  next gates: [vision investigation](SMOLVLA_VISION_INVESTIGATION.md).
- **2026-09-28, full 12-layer continuation:** Connected patch embedding, all 12
  packed blocks and connector with shared/ping-pong native buffers. Twenty
  rotating trials give 775.49 -> 531.62 ms against patched/unmasked fused B1.
  Three synthetic image variants retain full/parallel head parity. Numerical
  audit records extra differences versus fused FP16, including hidden-state
  outliers; closed-loop quality is not established. Consumer/MLP now slightly
  exceeds attention stage time, making two-view stage coordination the next
  meaningful question rather than merely increasing head parallelism.

- **2026-09-28, fair dual-view gate:** Twelve rotating trials with identical
  inputs and native boundaries: best fused mask1/mask2 view concurrency
  1103.42 ms; packed serial views 1056.27 ms; packed concurrent views 1056.44 ms.
  Including FP32/native input conversion and feature readback gives
  1115.65 / 1068.84 / 1050.58 ms. All are resident contexts, startup excluded.
  Swap/restore and schedule parity pass; fused/split numerical differences
  remain. NPU/DRAM clocks stay fixed. Concurrent packed views raise driver busy
  but not compute progress and have a worse p95 than serial packed views.
  The 4%-6% residual gain does not justify a large multi-view systems claim.
  Next gate is stage contention/complementarity, not more parallel threads.
- **2026-09-28, stage complementarity falsifier:** Twenty rotating trials on
  the actual final layer: two-view h4 Attention serial/parallel
  36.783/36.795 ms; consumer mask7 serial/parallel 38.102/38.074 ms.
  Cross-view Attention plus consumer serial/parallel is 37.379/37.314 ms;
  consumer mask1/mask3 overlap is worse (50.801/39.899 ms). All tested schedules
  retain consumer output parity. Pause the direct cross-view phase scheduler;
  open C12 as a bounded exact data-path test, not a new paper story.

- **2026-09-28, five-route reassessment:** FC1 work downgraded after scope and
  ceiling review. Prioritize complete native ten-step flow before naming a
  runtime: 8 ms host repacking is not the full input/transfer cost. Short-action
  CPU probes reopen only a geometry-specific question (C13); Torch FP16 still
  loses and FP32 timings are not same-precision proof. Preserve FP32 solver
  updates; FP16 unroll is not a free exact rewrite. Cloud remains a payload
  and winning-region audit with lossless transport, not an assumed winner.

- **2026-09-28, native ten-step strong control:** Implemented unchanged-graph
  persistent inputs with default and explicit dirty sync, plus logical FP16
  and native pass-through controls. The large gain over FP32 submission is
  absorbed by ordinary native pass-through (794.65 versus 793.24 ms); C09 is
  now a baseline, not the paper centerpiece. Strict storage-bit parity and a
  meaningful prefix invalidation check pass. Static graph projection remains
  independently untested. See [flow investigation](SMOLVLA_FLOW_INVESTIGATION.md).
- **2026-09-28, actual action-mask audit:** Captured all 32 action layers from
  the real checkpoint with synthetic prefix fixtures. Real causal self mask
  and GQA/layout increase fused CPU timings to 1.162/0.853 ms; only preprepared
  cross leaves a ~0.3 ms/layer clue against old NPU profiling. Full CPU single
  step velocity difference reaches 1.91e-6. C13 stays a bounded cross-block test
  with lowered expectations, not an automatic replacement main line. See
  [action attention investigation](SMOLVLA_ACTION_ATTENTION_INVESTIGATION.md).
- **2026-09-28, reading absorption and VLA-relevance correction:** Reject direct
  similarity-threshold target reuse/relaxed verification under current scope,
  but preserve approximate proposals behind a genuinely strict target verifier.
  AR speculation is not directly the deterministic ten-step flow protocol.
  Demote C13 from next main experiment. Reanalyze SDK conditioning rows: logical
  projection width 320 versus compiled reported 960, with 11.744 ms of static
  candidate operator time per old profile. Open C14 as a causal representation
  audit, not a new cache claim; C04 remains a payload gate, C15 only a ceiling
  check, C16 a compatibility reserve. These are distinct candidates, not four
  modules or an automatic fallback ladder. See
  [reading absorption](VLA_READING_ABSORPTION_2026-09-28.md).

## Evidence and closest systems work

- Local: [operator profile](../results/smolvla_libero/smolvla_vision_perf.txt),
  [two-view scheduling](../results/smolvla_libero/smolvla_camera_cores.json),
  [closed-loop stage report](smolvla_libero_closed_loop.md), and
  [research direction](SMOLVLA_RESEARCH_DIRECTION.md).
- Local: [B2 comparison](../results/smolvla_libero/smolvla_camera_batch2.json),
  [paired patch rewrite](../results/smolvla_libero/smolvla_vision_patch_pair.json),
  [head concurrency and reservation falsifier](../results/smolvla_libero/smolvla_attention_heads_reservation.json).
- [CoDL](https://www.microsoft.com/en-us/research/publication/codl-efficient-cpu-gpu-co-execution-for-deep-learning-inference-on-mobile-devices-2/): fine-grained mobile CPU/GPU coexecution.
- [HeteroInfer](https://arxiv.org/html/2501.14794v2): shape/order-aware mobile GPU/NPU tensor partitioning.
- [RoboECC](https://arxiv.org/html/2603.20711): model/hardware-aware edge-cloud VLA deployment.
- [ActionFlow](https://arxiv.org/html/2512.20276v1): VLA cross-request prefill/decode packing; a different scheduling axis, not a strawman.
- [Orca, OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf): direct prior art for selective batching.
- [Rammer, OSDI 2020](https://www.usenix.org/system/files/osdi20-ma.pdf): joint inter-/intra-operator execution planning.
- [LithOS, SOSP 2025](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/lithos_sosp25.pdf): kernel atomization and per-kernel right-sizing, with GPU control interfaces unavailable by default in RKNN.
- [MTS, SECON 2022](https://zhouzimu.github.io/paper/secon22-wang.pdf): shared weights versus graph fusion and temporary-memory growth.
- [SpotServe, ASPLOS 2024](https://arxiv.org/pdf/2311.15566): progress-preserving inference migration.
- [llada.cpp, 2026 preprint](https://arxiv.org/html/2606.13740): NPU-specific layout/mapping lifetimes; its dynamic token mechanisms do not establish exact continuous-flow speculation.
