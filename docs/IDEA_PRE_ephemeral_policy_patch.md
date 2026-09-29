# Idea pre v3: cloud-compiled, short-lived VLA policy patches

Status: research hypothesis, not a validated mechanism. 2026-09-25.

Superseded: the user has ruled out approximate policy changes. Retained as
research history only; do not pursue this route. The active exact-execution
investigation is `IDEA_PRE_exact_conditioning_runtime.md`.

## One-sentence problem

Under a constrained uplink, can one cloud VLA inference return a *short-lived
feedback policy* rather than one open-loop action chunk, so the robot can
correct for its own motion locally until visual or semantic evidence forces a
new cloud call?

Paper positioning: a new systems problem with a technique contribution. The
unit of offload is an ephemeral policy patch with an explicit validity region,
not a layer, frame, token budget, or fixed number of actions.

## Plain-language pain point and key observation

A robot cannot wait for a new image upload and cloud result after every tiny
movement, but simply replaying a long action chunk ignores what its joints
actually did. In many manipulation steps, the object and instruction have not
changed while the robot's own 8D proprioceptive state has. The cloud has
already spent the expensive vision/language computation to plan an action
chunk; throwing away all that work after one action seems wasteful. Yet a
full on-board VLA is not a fast fallback on the measured RK3588.

The proposed asymmetry is not generic image freshness. **Slow variables**
(task, object layout, phase) may remain valid across several motor steps,
while **fast variables** (joint/gripper state) are locally observable at
essentially no network cost. Ask whether the cloud can partially evaluate the
VLA around the current state and export a local feedback map. This is only
useful if the action function is sufficiently smooth within that region and
if departures can be detected before executing a wrong action.

## Concrete mechanism, inspired by Lever's phase-specific placement

At cloud call `k`, compute a nominal action chunk `A_k` and a small local
response model to changes in proprioception, e.g.

`A(s_k + ds) ~= A_k + J_k ds`.

The server can estimate `J_k` through batched finite differences or
automatic differentiation. Vision encoding and language/prefix work should be
shared across probes; only the conditional action-generation work is repeated.
Send `A_k`, a compressed/short-horizon `J_k`, the nominal state, and an
empirically calibrated trust region to the edge. The CPU applies the small
matrix update at every motor tick. The NPU is **not** asked to run the full
vision encoder continuously; it runs an optional compact, event-driven visual
guard when cheap proprioceptive checks become ambiguous. On guard failure,
contact/phase transition, trust-region exit, or patch expiry, upload a fresh
observation and request a new patch. The old patch may provide bounded
continuation while the request is in flight, subject to safety policy.

The Lever-like switch is thus: regular batched sensitivity probes on the
server GPU; irregular single-path correction on the edge CPU; edge NPU only
for selective visual verification. These roles are hypotheses to profile,
not claims that the NPU guard is already cheap or accurate.

## Why this is not merely a known trick

- Linear-feedback cloning and Jacobian controllers are old; a Jacobian is
  **not** the novelty. The research question is whether *runtime-compiled
  VLA policy patches* can convert cloud computation into useful edge-local
  feedback under measured bandwidth/deadline constraints.
- CloudEdgeVLA trains a persistent edge head using fresh local visual
  embeddings. ULAP trains a persistent small local policy. A policy patch is
  generated per cloud request from the current VLA and should require no
  separately trained local action model. This is a difference to test, not a
  guaranteed advantage.
- DREAM-Chunk and KeyStone already generate multiple candidate action chunks.
  Discrete branching or shared-prefix batch generation alone cannot be
  claimed here. A continuous local feedback map and a measured validity
  boundary are the distinguishing objects.
- Adaptive action chunking and VLA-Corrector already vary or validate the
  execution horizon. Replanning triggers alone cannot be claimed here.

## Main challenges and matching components

1. **Compilation cost:** Perturbing all state dimensions could erase the
   cloud-compute savings. Batch conditional probes with shared vision/prefix;
   test directional/low-rank Jacobians and short action windows. Report GPU
   occupancy, latency, and extra energy per patch.
2. **Validity:** The action map can be nonlinear near contact, gripper
   transitions, occlusion, or policy mode switches; fixed-image sensitivity
   does not model changing visual input. Calibrate per-phase trust regions on
   held-out rollouts, use conservative CPU state/contact guards, and invoke
   an NPU visual guard only where its measured latency improves decisions.
3. **Networking and timing:** A patch only helps if it covers enough valid
   actions to amortize upload and cloud generation. Use a deadline-aware
   renewal policy based on in-flight progress, uplink backlog, patch age,
   and measured trust-region margin. Never count a delayed cloud patch as
   valid for a newer observation without checking its anchor state.

## What the present measurements do and do not say

- SmolVLA on LIBERO produces a 50-action chunk but the tested configuration
  executes one action before replanning. This creates potential amortization,
  not proof that the remaining 49 actions are safe.
- The matched RK3588 path is about 3.236 s per inference; RTX 5060 compute is
  about 0.238 s without transport. RK vision encoding alone is about 1.749 s,
  so a continuously running local visual head is not a free fallback.
- A tiny horizon sweep over three seeds produced non-monotonic success and
  time outcomes. It is underpowered and its matched-noise implementation
  changes which noise sample is used after a skipped replan; it cannot support
  a causal claim about horizon quality.
- Naively batching full SmolVLA generation for 1/2/4/8 samples measured
  about 231/290/415/644 ms on the RTX 5060. This duplicates prefix work and
  is **not** a fair lower bound for shared-prefix methods. A diagnostic
  shared-prefix implementation failed a batch-output parity check, so its
  apparent denoising speed is not yet a valid result.

## Fastest falsification experiments, in order

1. **Local smoothness test before systems building.** On held-out LIBERO
   observations, fix images and denoising noise; perturb each normalized
   state dimension around the actual robot state. Compare full VLA actions
   with first-order predictions over realistic one-to-five-step deviations.
   Stratify free-space, approach, contact, and gripper transitions. Repeat
   when images advance, since fixed-image accuracy alone is insufficient.
   Kill or phase-restrict the route if error is not materially below naive
   open-loop action replay.
2. **Batch economics.** Implement correct shared-prefix conditional probes,
   pass output-parity tests against full inference, then measure end-to-end
   patch construction versus normal one-chunk inference. Vary rank, action
   horizon, and GPU load. Calculate actual downlink bytes including metadata.
   Kill if the patch cost exceeds the cloud calls it removes.
3. **Real link and asynchronous closed loop.** Start with a strong full-cloud
   JPEG baseline, not float32 image transfer. Use measured uplink traces and
   a simulator that advances during inference. Compare open-loop chunks,
   adaptive chunking, ULAP/CloudEdge-style local fallback if reproducible,
   and discrete candidate bundles. Report success, deadline misses,
   observation-to-action age, cloud calls, bytes, CPU/NPU/GPU energy, and
   wrong-patch activations.

Hard stop: if JPEG full-cloud meets control deadlines in realistic links, or
if a simple open-loop/adaptive chunk matches patch success at the same number
of cloud calls, do not build a larger runtime around this mechanism.

## Two alternative micro-mechanisms, lower priority

1. **Contingency packet:** Cloud GPU batches distinct *event-conditioned*
   action continuations; edge CPU selects one, NPU checks visual ambiguity.
   This is too close to DREAM-Chunk unless event diversity, true bandwidth
   benefit, and staged edge verification are empirically substantial.
2. **Hedged placement:** Begin a compressed-image upload and local partial
   VLA work concurrently, then switch at stage boundaries based on actual
   network progress. On this RK3588, local full inference is so slow that
   the hedge may be dominated; use only if a measured crossover exists.

## Closest primary references

- [Lever](../../../../llm-paper-reading/speculative/Lever-%20Speculative%20LLM%20Inference%20on%20Smartphones.pdf) (local PDF): phase-specific CPU/NPU placement.
- [CloudEdgeVLA](https://arxiv.org/html/2608.00569): cloud task features plus fresh edge vision.
- [VLA-ULAP](https://arxiv.org/abs/2609.18663): learned local action predictor with intermittent VLA calls.
- [DREAM-Chunk](https://arxiv.org/html/2606.18589): multiple candidates and within-chunk selection, including remote latency observations.
- [KeyStone](https://arxiv.org/html/2605.08638): shared-prefix batched candidate generation.
- [Linear Feedback Policy Cloning](https://openreview.net/pdf?id=BJl6TjRcY7): prior Jacobian-based local feedback idea.
- [VLA-Corrector](https://arxiv.org/abs/2607.01804): local detect-and-correct action horizon.

The literature check is preliminary and cannot establish novelty by absence.
