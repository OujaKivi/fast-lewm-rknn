# vla.cpp on RK3588: SmolVLA pilot

The tested upstream revision is `VinRobotics/vla.cpp` commit
`0644fc62d9bd404d7b6e19c9df0a1a3d2373ea65`. Its SmolVLA converter
accepted the same official `lerobot/smolvla_base` checkpoint used by this
project and produced a 777.3 MiB BF16 GGUF. The board build used four CPU
threads, `VLA_OCTO=OFF`, and the project's CPU backend. This upstream
revision does not expose an RK3588/RKNPU backend.

The `vla-bench` run used one 512x512 camera view, five language tokens,
zero warmups, one measured inference, and ten denoising steps. It uses
synthetic U8 pixels, placeholder token IDs, and deterministic noise, rather
than the *same values* as our PyTorch profile. The shape and checkpoint are
comparable, but this is not an action-quality parity test. A separate default
mode run with one warmup and three measured calls had 78,059 ms p50.

| RK3588 path | Vision | Multimodal prefix | 10-step denoising | End to end |
|---|---:|---:|---:|---:|
| PyTorch CPU, matched profile median | 3,968 ms | 6,469 ms | 29,309 ms | 39,759 ms |
| Existing RKNN hybrid, matched profile median | 879 ms | 6,462 ms | 516 ms | 7,872 ms |
| vla.cpp BF16, one phase-timed call | 44,911 ms | 5,227 ms | 24,589 ms | 74,769 ms |

The ggml CPU prefix is about 1.24x faster than PyTorch's prefix in this
shape-level pilot, but the complete vla.cpp CPU path is about 9.5x slower
than the existing RKNN hybrid. `vla.cpp` can accept precomputed image
embeddings, but its public prediction API does not expose the prefix K/V
cache needed by our RKNN denoising graph. Combining its prefix with our NPU
graphs would require an explicit cache-layout/precision bridge and numerical
validation; the table does not demonstrate that such a hybrid is faster.

We also tried the upstream `Q8_0` quantizer. Quantizing the vision tower
failed at model load (`unsupported dtype 8 for vit.blk.0.attn_q.weight`).
The default quantizer first failed on `mm.fc.weight`; locally exempting that
connector weight let it reach `vlm.blk.0.attn_q.weight`, which the SmolVLA
loader also rejected. Thus this revision's documented generic quantization
workflow is not directly usable with this converted SmolVLA checkpoint on
RK3588; no Q8 latency or quality claim is made.

The stock board build also hit an ARM64 C++ ABI link error for the two
`std::istream::seekg(offset, std::ios::beg)` calls in `src/models/smolvla.cpp`.
For this local pilot, using the equivalent `seekg(std::istream::pos_type(offset))`
overload allowed the build to finish. No upstream source is vendored into
this repository. The board needed `libzmq3-dev`, protobuf development tools,
and the `cppzmq` header; the Ubuntu 22.04 image did not provide `cppzmq-dev`.

**Conclusion:** vla.cpp is a useful C++ reference and its prefix implementation
suggests some CPU headroom, but using it wholesale does not accelerate the
current RK3588 deployment. The next meaningful optimization target is still
the approximately 6.46 s multimodal prefix, preferably with a validated
RKNN partition rather than a full runtime replacement.
