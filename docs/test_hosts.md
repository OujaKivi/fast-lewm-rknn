# Test hosts

These are lab inventory notes, not a claim that distributed inference is faster.
The host passwords below were supplied by the owner for this private lab
repository. SSH private keys remain outside the repository.

| Role | Address | Hardware | Current use |
|---|---|---|---|
| Edge target | `rk3588` / `192.168.77.2` | RK3588, 16 GiB shared RAM, 4 Cortex-A76 + 4 Cortex-A55, 3-core NPU | Fast-LeWM CPU/NPU; SmolVLA NPU-main-network LIBERO episode |
| GPU and simulator host | `192.168.77.10` | Intel i5-13490F (10 cores / 16 threads), RTX 5060 (8151 MiB VRAM), 16 GiB RAM, Ubuntu 24.04, NVIDIA driver 595.91.07 | LIBERO simulator, CUDA/CPU baselines, remote-policy client |
| Local Mac reference | Current workstation | MacBook Pro, Apple M5 Pro (15 CPU cores, 16 GPU cores), 24 GB unified memory | SmolVLA Metal/MPS remote-policy server |

Connect to the GPU reference with `ssh wang@192.168.77.10`. The login is
`wang` with password `66668888`; the existing developer SSH public key is
also authorized. The observed SSH host-key fingerprint is
`SHA256:dXq+TpZXnxAAclPhwK1LxB752QarOzrGdr6UfkZC1a4`.

The RK3588 login is `root@192.168.77.2` with owner-supplied password
`123456`. The current Mac connects through its SSH configuration; password
authentication has not been separately tested.

The GPU host had no outbound network access at the time of inventory. Model
weights and dependencies need to be staged from another machine. Benchmark
single-device latency, memory, and output quality before testing partitioning
or multi-machine collaboration; otherwise communication cost cannot be
separated from compute savings.

For the SmolVLA smoke, an isolated Python 3.12 environment is at
`~/smolvla-smoke/venv`. The checkpoint, tokenizer config, script, and offline
Linux wheelhouse are under `~/smolvla-smoke/`. This environment uses LeRobot
0.4.4, PyTorch 2.10.0+cu128, and CUDA 12.8. It does not modify the system
Python installation.

## Verified connections

These connections and runtime files were checked on 2026-09-24. The SSH
config and private keys remain outside the repository. The RK alias below
depends on the current Mac's `~/.ssh/config_rknn`; on a new workstation,
connect directly to `root@192.168.77.2` or recreate the alias.

```sh
ssh wang@192.168.77.10
ssh -F ~/.ssh/config_rknn rk3588
```

| Machine | Python | Model and runtime files |
|---|---|---|
| RTX/Linux | `/home/wang/smolvla-smoke/venv/bin/python` | `/home/wang/smolvla-smoke/models/smolvla_libero`, `/home/wang/smolvla-smoke/models/smolvlm_config`, `/home/wang/smolvla-smoke/eval_smolvla_libero.py` |
| RK3588 | `/root/.venvs/smolvla/bin/python` | `/root/models/smolvla_libero`, `/root/models/smolvlm_config`, `/root/smolvla-smoke/eval_smolvla_libero.py`; RKNN graphs listed below |
| Mac M5 Pro | `$HOME/.cache/fast-lewm-smolvla-venv/bin/python` | Hugging Face snapshot `HuggingFaceVLA/smolvla_libero` revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`; `$HOME/.cache/fast-lewm-smolvla-vlm-config` |

The RKNN files are:

```text
/root/smolvla-rknn-vision/vision_connector_fp16.rknn
/root/smolvla-libero-rknn/prefill32/prefill_addmask.rknn
/root/smolvla-libero-rknn/denoise32/denoise_step_addmask.rknn
```

The generated graphs are large and are **not** in Git. Source exporters,
conversion instructions, parity records, and the exact checkpoint protocol
are in [the LIBERO evaluation report](smolvla_libero_closed_loop.md).

## Matched-noise replay

Sync the current [`eval_smolvla_libero.py`](../scripts/eval_smolvla_libero.py)
to each remote host before replaying: client and server must agree on the
`--matched-noise` request format. Run only one simulator episode at a time.
The simulator is always on the RTX host; policy inference may be local or
remote. `HF_HUB_OFFLINE=1` and `MUJOCO_GL=egl` are set on that host.

```sh
scp scripts/eval_smolvla_libero.py wang@192.168.77.10:~/smolvla-smoke/eval_smolvla_libero.py
scp -F ~/.ssh/config_rknn scripts/eval_smolvla_libero.py rk3588:/root/smolvla-smoke/eval_smolvla_libero.py
```

For RK NPU-main-network inference, start this server on the RK board:

```sh
/root/.venvs/smolvla/bin/python /root/smolvla-smoke/eval_smolvla_libero.py serve \
  --model-path /root/models/smolvla_libero \
  --vlm-path /root/models/smolvlm_config --device cpu --threads 4 \
  --listen 0.0.0.0 --vision-rknn-path /root/smolvla-rknn-vision/vision_connector_fp16.rknn \
  --prefill-rknn-path /root/smolvla-libero-rknn/prefill32/prefill_addmask.rknn \
  --denoise-rknn-path /root/smolvla-libero-rknn/denoise32/denoise_step_addmask.rknn \
  --prefix-length 149 --layers 32
```

Then, on the RTX simulator host:

```sh
HF_HUB_OFFLINE=1 MUJOCO_GL=egl ~/smolvla-smoke/venv/bin/python \
  ~/smolvla-smoke/eval_smolvla_libero.py eval \
  --model-path ~/smolvla-smoke/models/smolvla_libero --device cpu \
  --host 192.168.77.2 --matched-noise \
  --output ~/smolvla-smoke/eval/matched_rk_npu_full.json
```

For RTX-local CUDA or CPU, omit `--host`, set `--device cuda` or `cpu`, and
add `--vlm-path ~/smolvla-smoke/models/smolvlm_config`. For Mac MPS, run
this from the repository root on the Mac:

```sh
MAC_MODEL="$HOME/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
"$HOME/.cache/fast-lewm-smolvla-venv/bin/python" scripts/eval_smolvla_libero.py serve \
  --model-path "$MAC_MODEL" --vlm-path "$HOME/.cache/fast-lewm-smolvla-vlm-config" \
  --device mps --listen 127.0.0.1 --port 47652
```

In a second Mac terminal, make a reverse SSH tunnel:

```sh
ssh -N -R 47651:127.0.0.1:47652 wang@192.168.77.10
```

The RTX client then uses `--host 127.0.0.1 --port 47651`. The RK listener
above binds to the lab LAN only; the Python connection protocol uses pickle
and must not be exposed to an untrusted network. For any non-isolated
network, bind to loopback and forward through SSH instead. No server or
tunnel is intended to remain running after a test. The four completed matched
episode records are in [`results/smolvla_libero/`](../results/smolvla_libero/)
under `matched_*_full.json`.
