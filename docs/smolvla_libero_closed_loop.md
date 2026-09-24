# SmolVLA LIBERO Closed-Loop Evaluation

The earlier `lerobot/smolvla_base` synthetic-input measurements are **not**
task evaluations: that checkpoint expects a 6D state and 6D action, whereas
LeRobot LIBERO supplies an 8D state, two cameras, and a 7D action. This test
uses the task-adapted
[`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
checkpoint at revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
The environment and policy preprocessing run on the RTX host. Remote
inference receives those processed observations and returns actions to the
same environment over a persistent authenticated connection.

## Protocol

- LeRobot 0.4.4 and `hf-libero` 0.1.4; `libero_spatial`, task 0, initial
  state 0, seed 1000, two 256x256 cameras, 8D state, 7D action.
- The checkpoint uses ten denoising steps and `n_action_steps=1`. Although it
  predicts a 50-action chunk, the policy replans after each executed action.
- A full episode ends on environment success or after at most 280 steps.
  One-step and two-step runs below only test closed-loop connectivity and
  cannot be interpreted as task failures or success rates.
- The inference metric times `select_action()` only: it excludes local
  preprocessing, action postprocessing, and simulator stepping. For remote
  paths it also includes request/response serialization and transfer.
  Episode wall time includes all stages and reset overhead; it is **not** a
  model-inference metric. The RK CPU/NPU comparison uses four PyTorch threads
  on the same board and the same first observation/seed.

## Results

| Deployment | Episode depth | Outcome | Mean inference / step | Episode wall time (incl. simulator) |
|---|---:|---|---:|---:|
| RTX 5060 CUDA | 76 steps | Success, 1/1 episode | 0.243 s | 21.3 s |
| Mac M5 Pro MPS, remote inference | 80 steps | Success, 1/1 episode | 0.496 s | 43.2 s |
| RTX host CPU | 70 steps | Success, 1/1 episode | 4.883 s | 345.3 s |
| RK3588, NPU vision + prefill + denoising, CPU auxiliary steps | 70 steps | Success, 1/1 episode | 3.240 s | 230.3 s |
| RK3588, four CPU threads, remote inference | 1 step | Action applied; no task verdict | 65.2 s | 66.3 s |
| RK3588, NPU vision + four CPU threads, remote inference | 1 step | Action applied; no task verdict | 59.0 s | 60.2 s |

There is **no RK3588 pure-NPU result** in this table. The successful RKNN
deployment offloads both image encoders, the 32-layer prefix transformer
(prefill), and the 32-layer action-expert transformer plus action-output
projection (denoising). On the RK3588 CPU, it still constructs the prefix
from image/language embeddings and state projection; builds masks and position
indices; packs/unpacks and transposes the 64 K/V cache tensors; computes the
action/time embedding MLP at each of ten denoising steps; and samples noise
and updates the action trajectory. The simulator host additionally performs
observation/action preprocessing, normalization, transport, and environment
stepping. Thus the NPU row means **NPU main networks plus CPU orchestration**,
not a CPU-free model or end-to-end NPU execution. An NPU-only deployment would
need new graphs/runtime integration for those remaining operations and has not
been implemented or measured.

The RTX CUDA/CPU, Mac, and RK CPU/vision-only paths also completed separate
two-step smoke tests.
Those earlier smoke tests did not seed the policy RNG and are retained only as
connection evidence. The full RTX CPU/CUDA, Mac, and task-shaped RK NPU
results show that these deployments can participate in and finish this task.
`1/1` is a capability check, **not** an estimated success rate. RK CPU and
NPU-vision-only execute the real observation-action-environment loop, but
their full task success is unmeasured; the RK pure-CPU full-episode run was
stopped at the user's request because each action took roughly a minute.
Even the complete RK NPU path runs at
only about 0.31 actions/s, so it is functional but not yet responsive control.
The simulator waits for inference between steps; task success does not imply
that a physical robot could tolerate the same wall-clock delay.

For the matched RK first step, vision took 7.848 s on CPU and 1.763 s on NPU,
while whole remote inference fell from 65.156 s to 59.033 s (1.10x). The
first environment actions had cosine 0.999979, MAE 0.00187, and maximum
absolute difference 0.00557. This is a local numerical check, not evidence
of equivalent task success.

The base and LIBERO checkpoints have exactly equal values for all 345 shared
VLM tensors, including all 198 vision/connector tensors, so the existing
vision RKNN graph is the correct one to reuse for this task checkpoint. The
LIBERO action expert has width 480 rather than the base model's 720 and runs
32 layers rather than 16. The first LIBERO observation has two images and 20
language tokens, giving a 149-token prefill with the state token. We therefore
exported separate FP16 RKNN prefill and denoising graphs for this checkpoint
and task; reusing the base model's 70-token graphs would have been invalid.

The 32-layer prefill graph produces all 64 K/V tensors in a median 252.7 ms;
the lowest cache cosine versus PyTorch is 0.999993 and the largest per-output
MAE is 0.00496. One 32-layer denoising step takes a median 106.8 ms and has
cosine 0.999999, MAE 0.00115 versus the float32 reference. The first complete
NPU action agrees with the RK CPU action at cosine 0.999927 and MAE 0.00326.
In the successful 70-step NPU episode, 122.2 s of inference was vision,
18.1 s prefill, and 76.3 s denoising, leaving about 10.2 s for CPU auxiliary
steps, transfer, and dispatch. Thus two-camera vision is now the largest
remaining stage (54% of inference). These graphs are fixed to task 0's 20
language tokens; other prompts need new shape handling or another export.

Raw records: [`results/smolvla_libero/`](../results/smolvla_libero/).
The Mac record's `remote:127.0.0.1:47651` endpoint is an SSH reverse port
forward to the Mac, not inference on the simulator host.

### Preprocessing and timing boundary

On the RTX 5060 path, the policy preprocessor moves input tensors to CUDA;
the language/state embeddings, masks, K/V tensors, action/time embedding,
and ten denoising updates inside `select_action()` therefore execute as CUDA
tensor operations (with Python control flow on the host). The environment
processors run before this move, while the final action is copied back to the
host for `env.step()`.

A repeated, seeded 76-step CUDA episode succeeded and separated the timing
stages: preprocessing (including device transfer) 0.109 s total, or **1.43
ms/action**; policy inference 19.392 s, or **255 ms/action**; action
postprocessing/copy 0.006 s, or **0.08 ms/action**; and simulator step 1.902 s,
or **25.0 ms/action**. The 22.448 s wall time additionally includes episode
reset and loop overhead. These are measurements for two virtual 256x256
cameras on the RTX host, not physical camera capture, ISP, network jitter, or
robot command latency. The original CUDA result above is a separate run;
explicit synchronization at the new stage boundary changes its timing
slightly. Pre/postprocessing are small in this setup and are recorded for
completeness, not added to the inference figure; simulator time is reported
separately and must never be counted as inference. Raw profile:
[`seeded_cuda_profile.json`](../results/smolvla_libero/seeded_cuda_profile.json).

For remote RK inference, preprocessing currently happens on the simulator
host and the already-processed tensors are transmitted to the board; that
transport is counted inside `inference_s_total`. A real RK deployment should
measure camera capture, decode/resize/normalization, model input preparation,
action postprocessing, and actuator output at their actual device locations
before deciding whether moving preprocessing changes end-to-end latency.

The evaluation harness is
[`scripts/eval_smolvla_libero.py`](../scripts/eval_smolvla_libero.py).
The simulator uses [LeRobot's LIBERO environment](https://huggingface.co/docs/lerobot/v0.4.4/libero).

## Reproduction

On a Linux simulator host with the LIBERO assets, the task checkpoint, and
LeRobot dependencies installed:

```sh
MUJOCO_GL=egl python scripts/eval_smolvla_libero.py eval \
  --model-path "$LIBERO_MODEL" --vlm-path "$SMOLVLM_CONFIG" \
  --device cuda --output results/smolvla_libero/local_cuda.json
```

To test a different inference device, start `serve` there with the **same**
checkpoint, a local SmolVLM config/tokenizer, and a private authentication
key. Use SSH port forwarding if the server is not on a trusted LAN. Then run
`eval --host HOST --port PORT --authkey KEY` on the simulator host. For the
RK3588 vision hybrid, add `--vision-rknn-path PATH --threads 4` to `serve`.
For the task-shaped full NPU path, add `--prefill-rknn-path PATH`
`--denoise-rknn-path PATH --prefix-length 149 --layers 32`. The server binds
to loopback unless `--listen` is explicitly changed.

The task-shaped graphs were exported with
[`probe_smolvla_prefill.py`](../scripts/probe_smolvla_prefill.py) and
[`probe_smolvla_denoise.py`](../scripts/probe_smolvla_denoise.py), both with
`--all-images --layers 32` and the exact task-0 prompt with a trailing
newline. Apply [`rewrite_smolvla_attention_mask_add.py`](../scripts/rewrite_smolvla_attention_mask_add.py)
before RKNN conversion. The compiled graphs are large (606 MB prefill,
215 MB denoising) and intentionally not committed. Their board/ONNX parity
records are in `results/smolvla_libero/`.
