# Test hosts

These are lab inventory notes, not a claim that distributed inference is faster.
Credentials and private keys must not be committed to this repository.

| Role | Address | Hardware | Current use |
|---|---|---|---|
| Edge target | `rk3588` (see local SSH config) | RK3588, 16 GiB shared RAM, 4 Cortex-A76 + 4 Cortex-A55, 3-core NPU | Fast-LeWM CPU/NPU benchmark; SmolVLA CPU smoke |
| GPU reference | `192.168.77.10` | Intel i5-13490F (10 cores / 16 threads), RTX 5060 (8151 MiB VRAM), 16 GiB RAM, Ubuntu 24.04, NVIDIA driver 595.91.07 | Hardware inventory only; no distributed inference result yet |

Connect to the GPU reference with `ssh wang@192.168.77.10`. The existing
developer SSH public key is authorized; no password is stored here. The
observed SSH host-key fingerprint is
`SHA256:dXq+TpZXnxAAclPhwK1LxB752QarOzrGdr6UfkZC1a4`.

The GPU host has no outbound network access at the time of inventory. Model
weights and dependencies need to be staged from another machine. Benchmark
single-device latency, memory, and output quality before testing partitioning
or multi-machine collaboration; otherwise communication cost cannot be
separated from compute savings.
