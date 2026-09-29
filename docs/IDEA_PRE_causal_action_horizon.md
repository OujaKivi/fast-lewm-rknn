# Idea pre: executed-prefix inference for causal flow-matching VLA

Status: hypothesis and experiment plan, not a paper claim. Updated 2026-09-25.

## 0. One-sentence position

Choose **causal flow-matching action-expert VLA**, beginning with the released
`HuggingFaceVLA/smolvla_libero` checkpoint. Its *generated* action horizon and
*executed* action horizon are different: the model generates 50 actions but our
closed-loop protocol executes only the first one. Because the action expert is
causal over action positions, computing the other 49 positions cannot change
that first action. The research question is whether making the accelerator
execute only the **causal cone of actions the controller will consume** creates
a meaningful quality-latency advantage, and whether the executed horizon can
be chosen without losing closed-loop responsiveness.

This is not a proposal for a generic graph portfolio, a universal edge-cloud
scheduler, or a claim that every flow-matching VLA admits this pruning. It
depends on a specific attention/dependency property and an actual mismatch
between prediction horizon and control horizon.

## 1. What the three reference papers teach

| Paper | Derivation chain | Taste worth emulating | Boundary for us |
|---|---|---|---|
| **Lever** | Flash-resident target weights make each verification expensive; mobile verification is not server-like parallelism -> larger speculative trees reduce I/O calls but enlarge verification compute -> optimize expected accepted tokens **per cycle latency**, not acceptance alone -> cost-aware tree, verification pruning, CPU/NPU placement. Its Table 1 shows 78.3%-93.3% I/O share for four target models; Fig. 5 shows acceptance and throughput diverging as tree budget grows. | Find a *reversal* of a familiar objective under a particular hardware regime. Model both useful work and its marginal system cost. Keep the target-model correctness boundary explicit. | Do not borrow token-tree speculation. Borrow the method of proving that the original objective (here, generating the full chunk or maximizing action throughput) is not the control objective. |
| **ActionFlow** | Single-robot OpenVLA seems to have no batching opportunities -> decompose one control stream into prefill plus serial decode micro-requests -> observe compute-bound prefill next to memory-bound decode and GPU bubbles -> pack stages from consecutive requests, then build a KV layout/fused kernels so packing really pays. Fig. 2's roofline and Table 1's naive-pipeline ablation connect insight to implementation. | Reinterpret the *internal dependency graph* of one request, not merely reduce FLOPs. Follow the idea through to data layout and synchronization; otherwise a clever schedule remains a slow prototype. | Its model is autoregressive OpenVLA, not a flow-matching action expert. Its result is FPS; for a robot, freshness and observation-to-action age must also be measured when using cross-time work. |
| **RoboECC** | VLA structural boundaries cause split-point latency discontinuities (Fig. 2); short action outputs expose network jitter rather than averaging it away (Fig. 3) -> jointly model structural cost and changing communication cost -> adjust the partition locally via duplicated boundary weights. | Use concrete counterexamples to break a generic deployment assumption. A structural discontinuity is more informative than saying hardware is heterogeneous. | Its method optimizes edge-cloud partitioning across architectures. It is not evidence that offload or generic split search is the distinctive problem for our local RK3588 deployment. |

**Their shared logic:** (1) locate a concrete mismatch between a model's
logical work and a platform/control objective; (2) show an observation or
counterexample that invalidates the usual optimization target; (3) exploit
an architectural invariant, not an arbitrary heuristic; (4) account for the
new overhead the intervention creates; (5) validate the *final* metric with
strong matched baselines. Lever's accepted length versus decode throughput,
ActionFlow's naive pipeline versus packed layout, and RoboECC's fixed split
versus bandwidth drift are all instances of this pattern.

There is also a lesson in restraint. ActionFlow's Algorithm 1 returns the
completed action sequence for request `t-(K-1)` while admitting the current
request `t`. Its reported FPS is therefore not an observation-to-action
freshness guarantee; the full pipeline delay and stale-observation effects
need their own measurement. RoboECC reports latency but does
not report task success in its main evaluation, reasoning that deployment
does not change model function. Our proposal should not inherit either gap:
we must report action age and paired closed-loop outcomes, even if the first
part of our optimization is mathematically output-preserving.

## 2. The local observation, derived rather than asserted

1. The locally cached, pinned LIBERO checkpoint config has `chunk_size=50`,
   `n_action_steps=1`, and `num_steps=10`. In the LeRobot policy,
   `select_action()` generates a chunk, enqueues only the first
   `n_action_steps`, and replans on the next observation. Our matched-noise
   closed-loop run uses precisely this protocol.
2. `embed_suffix()` creates one token for each noisy action position.
   `make_att_2d_masks()` uses cumulative block masks; the action positions
   are marked causal. The first action token can attend to the image,
   language and state prefix, but not to later action tokens. The suffix
   mask in our RKNN denoiser exporter is explicitly lower triangular.
3. Each Euler flow step updates action positions elementwise after the
   transformer predicts their velocities. If position `j` at every step
   depends only on positions `<=j`, future positions cannot feed back into
   an earlier action at a later flow step either.
4. Therefore, for the same prefix, initial-noise prefix, weights and
   arithmetic, the first `h` actions from a horizon-`h` run should equal
   the first `h` of a horizon-50 run. This is an **inference dependency**
   result, not a statement that executing `h` actions open-loop is safe.

Formally, let `F_t^H(x, c)` be the H-position velocity field at flow step
`t`, conditioned on prefix `c`, and let `P_h` select the first `h` positions.
The attention mask implies

```text
P_h F_t^H(x, c) = F_t^h(P_h x, c),       1 <= h <= H.
```

The Euler update `x_{t+1} = x_t - F_t(x_t,c)/S` commutes with `P_h`; induction
over the ten steps gives `P_h x_final^H = x_final^h`. The claim assumes the
exported backend implements the same mask and positional indices. FP16
rounding or different kernel shapes may yield small numerical differences,
so the *implementation* still needs a parity test.

Our one-observation MPS check with the released weights found first-action
MAE `0.00076` and max absolute difference `0.00230` between 50-position and
one-position execution. After warm-up, complete model calls were roughly
`0.55 s` versus `0.50 s` on that Mac. This is a smoke test, not a benchmark:
it uses synthetic images, one prompt, one noise sample, and no repeated
statistics. More importantly, it already refutes any naive inference that
discarding 49/50 action tokens means a dramatic **end-to-end** speedup.

On RK3588, the matched closed-loop episode spent about 54% of inference
time on the **model's two image-encoder calls**, then about 34% on ten
denoising steps and 8% on prefill. This is not a camera-capture, ISP, or
preprocessing measurement: LIBERO supplies virtual images, and preprocessing
currently runs on the RTX simulator host outside the timed inference region.
Even an infinitely fast denoiser could therefore save only
about one third of the currently measured action latency. A one-token NPU
graph may also use the accelerator worse than a 50-token graph. **Measured
RKNN latency, not token-count arithmetic, decides whether the idea lives.**

## 3. Candidate paper insight

> In a causal flow-matching VLA, the computation required to produce the
> controller's next `h` actions is the backward dependency cone of those
> `h` outputs, not the model's trained action horizon `H`. Matching the
> compiled inference graph to the *executed* horizon can remove unused
> action-token work without changing the actions that are consumed.

The distinctive part is the composition of **causal attention across action
positions**, **positionwise flow integration across denoising steps**, and
**closed-loop action consumption**. Remove any one of those and the exact
pruning argument fails. This is why the proposal does not automatically
apply to OpenVLA autoregressive decoding, a bidirectional DiT action head,
or a policy that executes its entire chunk.

The first system design is an **executed-prefix graph**: export RKNN action
expert graphs for a small set of horizons `h` (start with 1, 2, 4, 8, 16,
50), retaining the same prefix cache and the first `h` initial-noise values.
The runtime picks the smallest graph containing the actions it will actually
execute; it never computes a longer chunk merely to throw away its suffix.
This is a *shape-realization* problem because the NPU compiles fixed shapes.
We should profile actual graph latency, cache transfer, launch costs and
memory, not assume monotone speedup with smaller `h`.

If that primitive pays, the second design question is a control one:
**how many of the freshly predicted actions may safely be executed before
the next observation?** The policy can select an execution horizon `e`, then
generate only `h=e` actions. The decision should use a measurable freshness
or uncertainty signal *available before* the expensive action generation,
but no heuristic should be introduced before fixed-`e` curves show a real
quality-latency tradeoff. This is a nontrivial causality issue: a method
that chooses `e` from the complete new action chunk must compute all 50
actions first and cannot use executed-prefix pruning on that call. Novelty
would lie in coupling
control-horizon choice to *exactly reduced inference work* under the causal
cone, not in claiming adaptive action chunking itself is new.

## 4. What is not new, and what would be new

- **Not new:** executing different numbers of actions before replanning;
  [Adaptive Action Chunking (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf)
  already studies adaptive execution horizons. Its Algorithm 1 takes `N`
  already predicted full-`H` action chunks, computes per-position action
  entropy, and *then* selects how many actions to execute. It does not
  establish reduced generation work on this call. A useful distinction for
  us requires choosing `e` before full-chunk generation, not just setting
  `e` afterward.
- **Not new:** asynchronous execution and action continuity;
  [SmolVLA](https://arxiv.org/abs/2506.01844),
  [RTC](https://www.pi.website/download/real_time_chunking.pdf), and
  [FlashVLA](https://arxiv.org/abs/2608.27384) address related problems.
  [ActionFlow](https://arxiv.org/abs/2512.20276) targets a different,
  autoregressive action decoder.
- **Potentially new, subject to a full literature check:** a formal
  executed-prefix equivalence for a pretrained causal *flow-matching*
  action expert, realized as fixed-shape accelerator graphs, and coupled
  to the control horizon so that reduced open-loop commitment also reduces
  generation work. We have not established priority or publishability.
- **Not our claim:** generic token pruning, reducing denoising-step count,
  learned distillation, or faster remote inference. Recent
  [Coda](https://arxiv.org/abs/2609.21216) directly studies fewer flow
  steps; a paper centered on "10 steps to 5" would be poorly differentiated.

## 5. Three gates before committing to a CCF-A paper

### Gate A: dependency and parity (days, no board tuning)

Build a paired fixture from real LIBERO observations, with identical
prefixes and first-`h` noise for `h in {1,2,4,8,16,50}`. Compare **every
executed action coordinate** at the PyTorch velocity output for all ten
steps, the final normalized actions, and the postprocessed robot commands.
Run across tasks, prompts and initial states. Verify the masks in both the
Python implementation and the exported RKNN graph. The expected result is
near-numerical parity, not merely high action cosine. If the dependency
property fails for another checkpoint/attention mode, exclude it rather
than generalizing falsely.

### Gate B: real hardware payoff (one locked board session)

Export one-token and a few intermediate-horizon RKNN denoising graphs.
Benchmark warm and cold stage latency, complete `select_action()` latency,
P50/P95, CPU/NPU utilization, cache packing, graph load time, graph size,
memory and energy. Interleave `h=50` and `h=1` runs to control thermal drift.
Compare against the current full-50 RKNN path and a Python-only truncated
path. Use the same observation and noise; do not change step count, model
precision, camera inputs or graph placement between paired runs.

**Go/no-go:** continue only if a smaller graph materially improves
*complete* RK inference, not just the expert microbenchmark, and the gain
survives warm-state P95. We should pre-register the concrete minimum after
the stage profile and before looking at closed-loop outcomes. If graph
launch/weight traffic dominates, a one-token shape may save almost nothing:
then this is a useful negative result, not a CCF-A systems paper.

### Gate C: control-quality frontier (expensive, only after B)

Compare fixed execution horizons `e=1,2,4,8,16` and any later adaptive
policy against: full-50 generation with the same `e`, existing LIBERO
`e=1` protocol, fixed `e` baselines, and the published adaptive-chunking
idea where reproducible. Use paired LIBERO tasks/initial states and
matched-noise prefixes, while recognizing that different `e` values
produce different numbers of policy calls and therefore diverging noise
streams. Report task success with confidence intervals, completion steps,
observation-to-action age, action smoothness, P50/P95/P99 latency, energy
per executed action and deadline misses. The simulator waits during
inference; success there cannot establish physical control responsiveness.

For a publishable systems claim, include more than one task family and at
least one independent causal flow-action checkpoint or platform. A second
platform without the same causal action dependency would test portability
of the graph engineering, not generality of the insight.

## 6. Risks and decision tree

| Risk / reviewer objection | Decisive test | Consequence |
|---|---|---|
| "This is just setting `chunk_size=1`." | Show the dependency proof, existing policy output mismatch, graph-level implementation, real hardware costs and comparisons to output truncation and adaptive `e`. | If only a config change is needed and yields the same gain, do not claim a system contribution. |
| The RK NPU is weight/launch bound, so one token is barely faster. | Gate B shape scan and per-stage Amdahl accounting. | Stop this as a main paper idea; do not rescue it with unrelated optimizations. |
| Larger `e` improves amortized speed but hurts closed-loop success. | Paired Gate C success-latency frontier. | Stay with `e=1` exact pruning if the gain is enough; otherwise stop. |
| Prior adaptive chunking already shortens compute in this way. | Read its code and run matched implementation comparisons before drafting novelty claims. | Reframe or abandon rather than cite selectively. |
| Choosing `e` requires the full newly generated chunk. | Compare cheap pre-generation signals against full-chunk oracle decisions; include decision overhead and errors. | Keep a fixed executed horizon if the predictor costs more than it saves or hurts success. |
| Attention is not causal in a different flow VLA. | Verify mask and inter-step dependency model by model. | Limit scope to causal action experts; never claim all diffusion/flow VLAs. |
| Numerical parity on MPS does not survive RKNN FP16. | Intermediate velocity and final-action paired checks on the board. | Diagnose graph conversion; no closed-loop claim from cosine alone. |

## 7. Provisional paper skeleton

- **Problem:** trained action horizon `H` and controller-consumed horizon
  `e` are often mismatched; output-discard is not free on a small NPU.
- **Observation:** in causal flow action experts, the first `e` outputs
  are closed under both attention and the iterative flow update.
- **Method:** compile dependency-closed `e`-position action graphs and
  choose only graph shapes corresponding to actually executed actions.
- **Optional control mechanism:** choose `e` from a validated freshness
  signal, with the cost model using *measured* graph times, not token counts.
- **Evaluation:** exactness, stage and end-to-end hardware costs,
  closed-loop success/freshness frontier, thermal/energy behavior.

The paper should not be written until Gates A and B pass. The next concrete
artifact is a paired real-observation parity set plus an RK3588 horizon
latency table, not a broad runtime implementation.

## 8. Heterogeneous and edge-cloud boundary (not yet a contribution)

The current RK3588 path already puts vision, prefill and the denoising
transformer on NPU, with auxiliary embedding, cache layout and orchestration
on CPU. In the matched 68-step run, the residual after the three measured
NPU stages is about 0.145 s/action. Even eliminating that residual entirely
cannot fix a 3.236 s/action path; saying "CPU/NPU co-execution" is not a new
insight without a specific overlappable dependency and an end-to-end gain.

For cloud execution, the local RTX CUDA result of 0.238 s/action is a compute
reference **on the simulator host**, not a measured edge-to-cloud deployment.
The first experiment must compare (a) entire policy on RK, (b) complete
policy remotely with image transport, and (c) an actually justified cut at
the vision connector or prefix-cache boundary, all from the same edge-side
observation and under controlled bandwidth/jitter. A split is useful only if
its saved communication exceeds its edge compute plus extra transfer and
synchronization. With current RK vision around 1.75 s/action, processing
images on the edge before cloud offload is unlikely to beat full offload on
a good network; this is a hypothesis to test, not a result. Repeated network
round trips **inside** the ten-step denoising loop are particularly
unattractive; any viable split should transfer conditioning state once and
complete the loop on one side.

There is a possible future local-fast/remote-full fallback idea: let the edge
compute a dependency-closed short action prefix while the cloud computes a
longer chunk, and use the cloud result only if it arrives before the control
deadline. But today's edge path is not fast enough to justify calling this
"fast". It must remain outside the main idea until a measured local path
and realistic network-delay distribution create a nontrivial choice. If full
cloud dominates every practical network condition, or local fallback misses
the deadline, do not manufacture an edge-cloud contribution.

## Evidence and references

- Local [SmolVLA LIBERO protocol](smolvla_libero_closed_loop.md),
  [RKNN denoising exporter](../scripts/probe_smolvla_denoise.py), and
  [evaluation harness](../scripts/eval_smolvla_libero.py).
- Wang et al., *Lever: Speculative LLM Inference on Smartphones*, supplied
  PDF: `/Users/wangjiwei/llm-paper-reading/speculative/Lever- Speculative LLM Inference on Smartphones.pdf`.
- Dai et al., *ActionFlow: A Pipelined Action Acceleration for Vision Language
  Models on Edge*, supplied PDF:
  `/Users/wangjiwei/llm-paper-reading/vla/[DAC26]ActionFlow- A Pipelined Action Acceleration for Vision Language Models on Edge.pdf`.
- Zheng et al., *RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative
  Deployment for VLA Models*, supplied PDF:
  `/Users/wangjiwei/llm-paper-reading/vla/[IJCNN26]RoboECC- Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models.pdf`.
- The DAC26/IJCNN26 strings are filenames supplied with the PDFs; the
  local PDFs themselves were treated as the technical sources, not as
  independent verification of venue acceptance.
