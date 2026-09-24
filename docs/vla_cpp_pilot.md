# vla.cpp on RK3588: SmolVLA pilot

The tested upstream revision is `VinRobotics/vla.cpp` commit
`0644fc62d9bd404d7b6e19c9df0a1a3d2373ea65`. Its SmolVLA converter
accepted the same official `lerobot/smolvla_base` checkpoint used by this
project and produced a 777.3 MiB BF16 GGUF. The board build used four CPU
threads, `VLA_OCTO=OFF`, and the project's CPU backend. This upstream
revision does not expose an RK3588/RKNPU backend.

The initial `vla-bench` run used one 512x512 camera view, five language
tokens, zero warmups, one measured inference, and ten denoising steps. It
used synthetic U8 pixels and placeholder token IDs, so its shape and
checkpoint were comparable but its action values were not. A separate
default-mode run with one warmup and three measured calls had 78,059 ms p50.

| RK3588 path | Vision | Multimodal prefill | 10-step denoising | End to end |
|---|---:|---:|---:|---:|
| PyTorch CPU, matched profile median | 3,968 ms | 6,469 ms | 29,309 ms | 39,759 ms |
| Earlier RKNN hybrid with CPU prefill, matched profile median | 879 ms | 6,462 ms | 516 ms | 7,872 ms |
| vla.cpp BF16, one phase-timed call | 44,911 ms | 5,227 ms | 24,589 ms | 74,769 ms |

The ggml CPU prefill is about 1.24x faster than PyTorch's prefill in this
shape-level pilot, but the complete vla.cpp CPU path is about 9.5x slower
than the earlier RKNN hybrid and about 50x slower than the subsequent
1.50 s NPU-prefill path. `vla.cpp` can accept precomputed image
embeddings, but its public prediction API does not expose the prefill K/V
cache needed by our RKNN denoising graph. Combining its prefill with our NPU
graphs would require an explicit cache-layout/precision bridge and numerical
validation; the table does not demonstrate that such a hybrid is faster.

We also tried the upstream `Q8_0` quantizer. Quantizing the vision tower
failed at model load (`unsupported dtype 8 for vit.blk.0.attn_q.weight`).
The default quantizer first failed on `mm.fc.weight`; locally exempting that
connector weight let it reach `vlm.blk.0.attn_q.weight`, which the SmolVLA
loader also rejected. Thus this revision's documented generic quantization
workflow is not directly usable with this converted SmolVLA checkpoint on
RK3588; no Q8 latency or quality claim is made.

An additional exact-input parity check used the same resized float32 image,
token IDs, zero state, and PyTorch-seeded initial noise as the cross-device
profile. The complete vla.cpp path reached only `0.948093` action cosine
and `0.20120` MAE versus the i5 PyTorch CPU result. Feeding the exact
PyTorch vision connector output through vla.cpp's public
`precomputed_img_emb` input raised action cosine to `0.999991` and reduced
MAE to `0.00249`; vla.cpp then took 29,810 ms, including 5,216 ms of
prefill and 24,556 ms of denoising. This isolates the large parity gap to
the vla.cpp vision path for this checkpoint/input, not its prefill or action
expert. The two action arrays and the fixture generator are saved as
`results/smolvla_vla_cpp_rk3588_actions.npy`,
`results/smolvla_vla_cpp_rk3588_precomputed_vision_actions.npy`, and
`scripts/prepare_vla_cpp_fixture.py`. No task-success claim follows from
these synthetic action comparisons.

The stock board build also hit an ARM64 C++ ABI link error for the two
`std::istream::seekg(offset, std::ios::beg)` calls in `src/models/smolvla.cpp`.
For this local pilot, using the equivalent `seekg(std::istream::pos_type(offset))`
overload allowed the build to finish. No upstream source is vendored into
this repository. The board needed `libzmq3-dev`, protobuf development tools,
and the `cppzmq` header; the Ubuntu 22.04 image did not provide `cppzmq-dev`.

**Conclusion:** vla.cpp is a useful C++ reference and its prefill implementation
suggests some CPU headroom, but using it wholesale neither accelerates the
current RK3588 deployment nor preserves this checkpoint's action output
through its vision path. The former 6.46 s multimodal prefill bottleneck has
since been addressed by the validated RKNN partition documented in
`docs/smolvla_rknn.md`.
