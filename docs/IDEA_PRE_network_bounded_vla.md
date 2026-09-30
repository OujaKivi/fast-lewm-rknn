# Idea pre v2: control-deadline-aware VLA across edge accelerators and a constrained link

Status: proposed main route, not an established result. 2026-09-25.

2026-09-30: current scope and execution gates are in the
[main-line plan](MAINLINE_EXECUTION_PLAN_2026-09-30.md). This historical proposal
is not revived: no cached views, lossy payload, freshness premise or changed
control protocol. Same-observation collaboration must beat strong endpoints.

2026-09-28 status correction: historical proposal, not the current main route.
Cached-view, token reduction, lossy transfer and freshness-dependent claims
are outside the current scope. Only the measured same-observation lossless
payload/placement feasibility gate is retained; see
[current reassessment](IDEA_REASSESSMENT_2026-09-28.md).

## Scope and paper type

Technique paper. Focus on **multi-camera, flow-matching action-expert VLA**,
starting with `HuggingFaceVLA/smolvla_libero`. Do not claim a universal VLA
runtime or use Fast-LeWM as a second, unrelated application. The system
boundary is a robot with a CPU/NPU, a constrained wireless uplink, and a
GPU server. The existing RTX 5060 machine can stand in for the server in
controlled experiments; this is not yet a measured public-cloud deployment.
With current hardware we can validate an end-to-near-edge link; a genuine
three-tier end/edge/cloud claim would need a separate remote cloud resource
and measured links on both hops.

## Pain point, in one paragraph

A remote VLA can compute quickly, but it must act on camera observations that
cross a limited uplink. Sending every camera frame can queue or delay the next
decision; sending fewer observations can make the returned action stale. Full
local inference avoids the network but is slow on the current RK3588. For a
robot, the failure is not merely a larger average inference number: an action
may arrive after the moment it was computed for. The system must decide which
visual information to move, which device should process it, and whether a
new result will arrive before the controller needs it.

This differs from RoboECC's story about the best layer split moving as
bandwidth changes. A different best split is a system symptom, not the
robotics-specific pain point. **Observation age at action execution and
deadline misses** are the primary quantities here.

## Existing evidence and the first counterexample we must respect

- The matched RK3588 episode took 3.236 s/action inside `select_action()`:
  1.749 s for two model image encodings, 0.256 s prefill, 1.086 s ten-step
  denoising, and about 0.145 s residual. "Vision" here means model encoding,
  not physical camera capture/ISP. Current preprocessing runs on the RTX
  simulator host outside that inference timer.
- The RTX CUDA reference took 0.238 s/action with simulator and inference
  on the same host. This **does not include robot-to-server transport**.
- Two 256x256 RGB views are about 1.5 MiB as float32 model inputs, 0.375 MiB
  as uncompressed uint8, and potentially much smaller as compressed images.
  The connector produces 128 visual tokens total; at width 960 its nominal
  FP16 payload is about 0.234 MiB before protocol overhead. Thus "send
  features instead of images" is not automatically a communication win.
  A fair full-cloud baseline must resize and compress on the robot; it must
  not transmit gratuitous float32 tensors.

The immediate threat to this idea is clear: if full-cloud JPEG transport plus
GPU inference beats every edge-vision split under realistic constrained
traces, a model-splitting contribution is not there. Network restriction
cannot be chosen after seeing results merely to make a favored split win.

## Model-specific insight to test

SmolVLA is not one undifferentiated chain of transformer layers. Its two
camera views are independently encoded before a shared prefix, while the
action expert reuses that prefix through ten denoising iterations. A cut
*inside* the iterative loop would require repeated synchronization or
state transfer; a cut before the loop can send conditioning state once.
Likewise, the two camera branches need not be placed or transmitted in the
same way if their update value and deadlines differ.

The candidate insight is:

> For network-constrained, multi-camera flow VLA, the useful scheduling unit
> is a **camera branch plus a complete denoising loop**, aligned to the next
> control deadline, not an arbitrary layer boundary or a bandwidth-only
> token quota.

The specific hypothesis is that a small number of architecture-valid plans
can form a non-dominated frontier under varying observation freshness,
uplink capacity, edge NPU availability, and execution-window length. Plans
include full-cloud compressed images, per-camera edge-NPU feature production
with compressed feature transfer, and full local execution. The server should
complete prefill and the entire denoising loop on one device unless a new
measurement proves a finer cut worthwhile.

This is **not** yet a claim that feature transfer helps. The edge vision
encoder currently takes about 1.75 s for both views. A compressed feature
path must save more communication/cloud work than its edge compute cost, or
hide that cost behind already executing actions. Its actual payload,
compression/parity, and control-quality effect must be measured.

## Proposed mechanism, conditional on the evidence gate

1. **Dependency-aware execution plans.** Compile/implement only legal
   camera-branch and whole-loop placements. Keep an explicit plan table of
   stage latency, transfer bytes, memory, energy and numerical quality on
   CPU, RK NPU and server GPU. Avoid arbitrary transformer-layer search.
2. **Progressive camera transport and overlap.** Where useful, CPU image
   preparation and uplink of one view overlap with NPU encoding of another;
   the server starts any independent vision work as soon as its input
   arrives. Cloud prefill begins only when the selected fresh/cached views
   are available. Reusing a cached view is a quality decision, not a free
   system trick; it needs paired task validation.
3. **Deadline-aware plan admission.** At each observation, use measured
   conditional latency distributions, current uplink backlog, NPU state,
   and time until the next action is needed to select a plan. Optimize
   probability of a *fresh* action arriving by its deadline under task
   quality and energy constraints, not merely predicted mean inference time.
   A stale/late cloud result must not silently replace the action for a
   newer observation.

The three components should exist only if the first experiments show
crossing plans and useful overlap. Otherwise, this would be an elaborate
implementation of a dominated path.

## Prior-work boundary

- [RoboECC](https://arxiv.org/abs/2603.20711) searches model split points
  and adjusts them with bandwidth. We would not claim generic dynamic
  partitioning as new.
- [ComVLA](https://arxiv.org/html/2609.07838) already adapts visual-token
  count to channel capacity using language-guided importance. A channel
  budget or visual-token pruning alone is not our novelty.
- [RAPID](https://arxiv.org/html/2603.07949) uses kinematics to choose
  edge/cloud behavior and exploit step-wise redundancy. A phase trigger
  alone is not our novelty.
- [CloudEdgeVLA](https://arxiv.org/html/2608.00569) and
  [VLA-ULAP](https://arxiv.org/abs/2609.18663) already pair delayed cloud
  computation with local action generation. A generic local fallback or
  slow-cloud/fast-edge architecture is not our novelty.
- [Adaptive Action Chunking](https://arxiv.org/html/2604.04161) already
  changes how many actions are executed. We may use execution-window length
  as an input, but cannot present adaptation itself as new.

The only defensible differentiation would be a *measured*, model-specific
interaction among independent camera branches, constrained communication,
whole-loop placement, and control deadlines. A literature review beyond
these closest works is still required before a novelty claim.

## Experiments in decision order

### Gate 0: does a nontrivial design space exist?

From the same edge-side observations, measure five complete paths under
interleaved board states: (A) full local CPU/NPU; (B) full-cloud with resized,
compressed images; (C) each camera encoded on RK NPU then feature payload
sent for cloud prefill/action; (D) one camera cloud-encoded and the other
edge-encoded; (E) cached-view variants. Use actual byte counts including
serialization, codec cost, RTT, and graph load. Measure P50/P95/P99 of
observation-to-action age, not only `select_action()` time. Quantize or
compress features only after measuring action-level parity.

Collect or replay real Wi-Fi/hotspot traces, plus controlled sweeps of
uplink rate, RTT, jitter, loss and outages. Cover good, crossover and poor
regimes; publish the complete range rather than only a favorable point.
Pre-register a stop condition: if compressed full-cloud dominates all legal
splits and edge fallback never meets a meaningful deadline, stop this
paper route. If only very low, implausible bandwidth makes a split win,
report that as a limitation instead of calling it a general solution.

### Gate 1: prove the scheduling mechanism, not just a lucky plan

Compare best static plan, full cloud, local-only, RoboECC-style cost-based
split, ComVLA-style channel token budget where reproducible, and an offline
oracle. Ablate per-camera placement, transfer representation, overlap,
deadline estimation and stale-result rejection. Report quality, energy,
bytes, camera/NPU/GPU occupancy and control-deadline misses. If a simple
"always full cloud" or "always local" policy matches the proposed runtime,
there is no mechanism claim.

### Gate 2: closed-loop validity

The current LIBERO harness is synchronous: the simulator waits while
inference runs. Thus its success rate does **not** test whether wireless
delays make actions stale. Add a fixed-rate/asynchronous simulator protocol
that advances the environment during inference and applies a declared
hold/previous-action/safe-stop policy when no fresh action is ready; then
evaluate paired tasks and initial states. Include a physical robot only
after this protocol is sound. Report success and action age together; a
high simulated success rate under a paused simulator is not sufficient.

## Provisional claim and kill criteria

Potential claim, only if Gates 0-2 pass: *A dependency-aware execution and
transport runtime for multi-camera flow VLA meets more control deadlines
with fresher actions than full offload or layer-centric splitting under
measured constrained links, without reducing task success.*

Kill the claim if any of these occur: no realistic plan crossover; edge
vision cost overwhelms communication savings; missing/cached camera views
damage task success; asynchronous deadlines remain unmet even after
overlap; or the gains reduce to channel-aware token pruning or a generic
threshold router already covered by prior work. The earlier exact
action-prefix idea is an optional optimization within a viable plan, not
the center of this paper.

## Local evidence

- [Matched LIBERO rollout](../results/smolvla_libero/matched_rk_npu_full.json)
  and [timing/protocol report](smolvla_libero_closed_loop.md).
- [Current server/evaluation harness](../scripts/eval_smolvla_libero.py):
  synchronous simulator, two processed camera tensors, remote request path.
- [Earlier action-prefix pre](IDEA_PRE_causal_action_horizon.md), retained
  as a separate candidate rather than merged into an all-purpose system.
