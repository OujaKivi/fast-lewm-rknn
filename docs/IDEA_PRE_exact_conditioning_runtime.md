# Exact request-scoped conditioning for SmolVLA

2026-09-25. Supporting optimization; no longer the proposed paper centerpiece.
See `SMOLVLA_RESEARCH_DIRECTION.md` for the current evidence and research gates.
Supersedes the approximate local-policy-patch direction. Keep all observations,
action tokens, denoising steps, weights, and the execution protocol unchanged.

2026-09-28 update: the unchanged ten-step native loop is implemented and strictly
bitwise checked against the existing graph. Resident inputs take 793.24 ms, but
ordinary prepacked native pass-through takes 794.65 ms. The input-lifetime speed
claim is therefore rejected as a paper centerpiece. Follow-up cold/warm graphs
are now tested: ordinary consumer-ready caching cuts full action-flow latency
747.77/749.43 -> 692.65/693.32 ms with preparation charged and tested step bits
identical. Compact/grouped paths lose in the layer-1 fragment. This remains a
strong baseline, not a paper centerpiece. See
[conditioning validation](SMOLVLA_CONDITIONING_VALIDATION.md); the proposal below is historical.

## The small contradiction

The deployed SmolVLA caches VLM prefix K/V, but the cross-attention expert
projects those same tensors again during each denoising iteration. The cache
boundary is before, rather than after, the last query-independent transform.
Separately, the board wrapper repacks all prefix tensors before each call.
The logical request has persistent conditioning, while its execution interface
treats each denoising step as a stateless invocation with fresh inputs.

Source inspected: the installed board `smolvlm_with_expert.py`, especially
`forward_cross_attn_layer`. Current upstream also performs the expert K/V
projections inside that function. Local RKNN invocation and repeated layout
conversion are in `scripts/eval_smolvla_libero.py`, lines 95-113.

For a fixed request, fixed weights, and inference mode:

`K_expert[l] = linear_K[l](K_prefix[l])`

`V_expert[l] = linear_V[l](V_prefix[l])`

Neither depends on the noisy action or denoising time. Cache these exact
results for cross-attention layers only. Self-attention action K/V and all
attention outputs remain dynamic and must be recomputed. Invalidate at every
new prefix/request; this does not reuse old observations across robot steps.

## A possible systems mechanism, beyond ordinary caching

Partition execution by **data lifetime**, not merely by model layer:

1. A preparation path produces consumer-ready, native-layout conditioning.
2. A recurrent NPU graph consumes that persistent conditioning and only
   exchanges changing action state with the host.
3. Optionally, the first denoising iteration runs the original graph while
   CPU prepares the projected K/V for later iterations. Once preparation and
   device synchronization finish, switch to the recurrent graph. If not ready,
   keep using the original graph; no prediction or approximate action is used.

This cold-to-warm transition overlaps preparation with an already required
iteration. It deliberately duplicates some first-iteration work to remove
later work from the critical path. It must not be confused with reducing the
iteration count, reusing approximate activations, or compiling new weights per
request. The two graphs can be compiled ahead of time.

Let P be CPU preparation time, D0 original-step latency, Dw warm-step latency,
and S switching plus transfer cost. For ten iterations, the idealized paths are:

- CPU preparation first: P + 10 Dw.
- Cold/warm overlap: max(P, D0) + 9 Dw + S.

When P <= D0, overlap only beats that serial path if P > D0 - Dw + S.
Real bandwidth interference and readiness must be measured. This comparison
alone is insufficient: NPU-fused preparation can be a stronger baseline.

## Board probes completed

### 1. Host layout conversion

Using the existing 32-layer, 149-prefix-token, 50-action-token exported graph:

- 64 prefix tensors, each `[1,149,5,64]`, total 12,206,080 FP32 bytes.
- Dynamic suffix input: 96,000 bytes.
- Layout conversion median: 8.204 ms, over 20 samples.
- Repack plus original NPU call: median 114.180 ms, p95 132.409 ms.
- Reuse already packed host arrays plus original NPU call: median 107.602 ms,
  p95 123.001 ms. Alternating comparison order, 20 samples per mode.
- Maximum output difference: 0.

This is a host-side optimization, not evidence of resident NPU inputs or
eliminated device transfers. Both modes still call the standard RKNN API with
all input tensors. The raw tensor-byte ratio is not a measured DRAM traffic
ratio. These numbers are fixed-input microbenchmarks, not full rollouts.

### 2. Exact projection hoisting

The real LIBERO checkpoint has 16 cross-attention layers and 32 independent
K/V projections. Each uses `[1,149,320]` input and `[320,320]` weights.

- Prepared projection outputs occupy 6,103,040 FP32 bytes.
- RK CPU, four Torch threads: one set of projections median 26.472 ms.
- The probe asserts each projection input is unchanged across two distinct
  action suffixes. All 64 assertions passed.
- The complete 32-layer expert outputs with hoisted versus original
  projections were bitwise equal for both FP32 CPU test inputs.

This validates the tested operation rewrite, not NPU numerical parity across
devices, a full ten-step trajectory, or closed-loop task equivalence.

### 3. CPU preparation can overlap an original NPU step

Ten alternating trials per mode, on the board, with both outputs checked:

- Serial CPU preparation plus one original NPU step: median 153.722 ms,
  p95 168.979 ms.
- Concurrent CPU preparation and one original NPU step: median 110.831 ms,
  p95 125.135 ms.
- CPU prepared tensors were bitwise unchanged; NPU output max difference 0.

These are joint-completion measurements. They demonstrate available overlap,
but not a warm-graph speedup: the NPU still executes the original graph and
does not consume the CPU-prepared projections in this experiment. Separate
CPU timing and the serial/concurrent trials have different execution contexts;
do not subtract their medians to infer exact device costs.

Records: `results/smolvla_libero/kv_lifetime_probe.json` and
`results/smolvla_libero/static_projections_overlap_probe.json`.
Reproduction scripts: `scripts/probe_smolvla_kv_lifetime.py` and
`scripts/probe_smolvla_static_projections.py`.

## Novelty audit and strongest counterexamples

**Exact K/V caching is not new.** Diffusers already supplies lossless text K/V
caching, and encoder-decoder Transformers have long cached cross-attention
projections. Likewise, RKNN provides native/zero-copy memory APIs. The current
findings can be useful implementation improvements without being a paper.

The hypothesis worth pursuing is narrower: whether request-scoped preparation,
native-layout residency, and a cold/warm heterogeneous transition jointly
outperform the strongest single-device exact execution under mobile hardware
constraints. Novelty is not established by this preliminary search.

Mandatory baselines, all with identical model semantics:

1. Hoist projection and layout conversion once, without heterogeneous overlap.
2. Fuse expert K/V projection into NPU prefill and pass native buffers onward.
3. Have the first NPU denoising graph emit its already computed projections,
   then reuse them in the warm graph; avoid duplicating work on CPU.
4. CPU-prepared cold/warm overlap with persistent bound inputs.

If baseline 2 or 3 wins consistently, drop the CPU-overlap novelty claim.
Measure graph-switch latency, two-context weight memory, native layout/stride
compatibility, cache coherency, and total DRAM traffic. Zero-copy is not zero
memory traffic, and a persistent binding does not guarantee on-chip residency.

## Next evidence gate

Export a warm denoising graph whose cross-attention inputs are already
projected. First validate it against the same original floating-point model
for multiple prefixes, action inputs, and all ten steps. Then compile and test
NPU parity using the existing numerical-error standard, without weakening it
to accommodate the optimization. Compare the four execution plans above.

Only after a repeatable end-to-end advantage should this become the main
paper direction. Denoising is about 1.086 / 3.236 of the existing end-to-end
inference time, so even removing it entirely would cap speedup at about 1.51x
on that measured path. Projection-only improvements can save much less time
than eliminating the entire denoising stage.

The method may apply to VLMs with immutable cross-attention conditioning, but
does not automatically apply to evolving self-attention K/V or permit reusing
features from changed images.

## Primary sources

- [SmolVLA upstream implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
- [Diffusers exact text K/V cache](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/text_kv_cache.py)
- [Rockchip native-memory example](https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_api_demo/src/rknn_create_mem_demo.cpp)

Live upstream sources may change; conclusions about our deployed version
come from the installed board implementation and the saved probes above.
