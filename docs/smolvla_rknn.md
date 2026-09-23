# SmolVLA RK3588 NPU pilot

This pilot moves the image encoder/connector and the cached 16-layer action
denoising step to the RK3588 NPU. The language prefix and denoising suffix
embedding stay in PyTorch on four Cortex-A76 cores. The checkpoint is the
official `lerobot/smolvla_base`, with LeRobot 0.4.4, RKNN Toolkit2/Lite2
2.3.2, and NPU driver 0.9.8.

## Reproduce

Set `MODEL_PATH` to the local SmolVLA checkpoint directory and `VLM_PATH`
to a local SmolVLM2-500M config/tokenizer directory. The full SmolVLA
checkpoint already contains the VLM weights; no second VLM weight download
is needed. Set `OUTPUT_DIR` to an untracked working directory.

```sh
python scripts/export_smolvla_vision.py \
  --model-path "$MODEL_PATH" --vlm-path "$VLM_PATH" \
  --output-dir "$OUTPUT_DIR"

docker run --rm -v "$OUTPUT_DIR:/work" -v "$PWD:/repo:ro" \
  rknn-toolkit2:2.3.2 python /repo/Fast-LeWorldModel/convert_to_rknn.py \
  --onnx /work/vision_connector.onnx \
  --out /work/vision_connector_fp16.rknn --dtype fp16
```

The exporter uses static `512x512` position indices. It checks that the
rewrite exactly matches the original PyTorch vision output before exporting.
The compiled 227 MB RKNN artifact is intentionally not tracked in Git. On
the board, use `scripts/test_smolvla_vision_rknn.py` with the exported
`input.npy` and `reference.npy`, then
`scripts/eval_smolvla_hybrid_board.py` with the checkpoint and RKNN file.
The tested RKNN file has SHA-256
`fc447ae1a83b81fdc958e1bdc2519cfb509adf793fed35474e6c1cb92fc597d0`.

The denoising graph is exported separately. It accepts one suffix embedding
`[1,50,720]` and 16 pairs of cached prefix K/V tensors
`[1,70,5,64]`; the board caller transposes the four-dimensional caches to
NHWC before RKNNLite inference. The exported graph fixes the mask and
position IDs for 70 fully valid prefix tokens, so a different prompt length
or padded prefix needs a new export.

```sh
python scripts/probe_smolvla_denoise.py \
  --model-path "$MODEL_PATH" --vlm-path "$VLM_PATH" \
  --output-dir "$OUTPUT_DIR/denoise"

python scripts/rewrite_smolvla_attention_mask_add.py \
  --input "$OUTPUT_DIR/denoise/denoise_step.onnx" \
  --output "$OUTPUT_DIR/denoise/denoise_step_addmask.onnx"

docker run --rm -v "$OUTPUT_DIR/denoise:/work" -v "$PWD:/repo:ro" \
  rknn-toolkit2:2.3.2 python /repo/scripts/convert_smolvla_denoise_rknn.py \
  --onnx /work/denoise_step_addmask.onnx \
  --output /work/denoise_step_addmask.rknn
```

Validate the ONNX graph with `scripts/test_smolvla_denoise_onnx.py` before
compiling and the board graph with `scripts/test_smolvla_denoise_rknn.py`
using `--layout nhwc`. The validated 203 MiB denoising RKNN file has SHA-256
`bf7649c8a8d5c087bf6db5a22e832e242c3b5427d06e4b723ddb5091851fc1a8`.

## Measurements

### Matched Cross-Device Stage Profile

All paths below use the same synthetic image, instruction, state, initial
noise (seed 42), checkpoint, and ten denoising steps. Stage timers synchronize
MPS/CUDA before and after each measured operation. The prefix column is the
cached image/language transformer pass after vision embedding; denoising
includes all ten steps. `Other` is measured end-to-end time minus the three
timed stages. Values are medians of warmed runs in milliseconds; individual
stage medians need not sum exactly to the end-to-end median. The final column
is the cosine similarity of the final `[1,50,6]` action chunk against the
i5-13490F pure-CPU output, **not task success rate**.

| Device / path | Vision | Prefix | 10-step denoising | Other | End to end | Action cosine |
|---|---:|---:|---:|---:|---:|---:|
| i5-13490F CPU | 654.7 | 274.3 | 1,377.9 | 3.1 | 2,309.1 | 1.000000 |
| RTX 5060 CUDA | 31.2 | 8.2 | 80.7 | 2.1 | 122.2 | 0.999991 |
| Mac MPS | 68.6 | 15.7 | 142.8 | 4.5 | 232.6 | 0.999885 |
| RK3588 CPU | 3,968.2 | 6,469.0 | 29,308.6 | 13.2 | 39,759.1 | 0.999988 |
| RK3588 NPU vision + CPU denoising | 878.6 | 6,463.7 | 29,274.5 | 13.1 | 36,629.9 | 0.999850 |
| RK3588 NPU vision + NPU denoising | 878.7 | 6,462.0 | 515.7 | 14.0 | 7,872.0 | 0.999853 |

The RK3588 accelerated path is 5.05x faster than its CPU path in this
matched test. Its prefix pass now accounts for about 82% of total latency.
The breakdown and action arrays are saved in
`results/smolvla_profile_*.json` and `results/smolvla_profile_*.npy`;
re-run with `scripts/profile_smolvla_stages.py`. The board CPU and
NPU-vision/CPU-denoising rows have two timed repeats; the board full-NPU,
Mac MPS and i5 CPU rows have three; RTX CUDA has five. These are synthetic
smoke inputs, not task episodes.

- [Vision graph: 20 warmed runs](../results/smolvla_rk3588_vision_rknn.json):
  median 872 ms, p95 877 ms, output cosine 0.999825, MAE 0.0547.
- [Original denoising graph](../results/smolvla_rk3588_denoise_original_mask.json):
  cosine 0.9556, MAE 0.200 versus float32 PyTorch after correct NHWC cache
  layout. Layer-one taps showed that unmasked logits matched closely, but
  masked logits did not: masked entries were finite scores instead of the
  required suppressing values before Softmax. The float32 ONNX and RKNN
  simulator both matched PyTorch, isolating the defect to the board-compiled
  Boolean `Where`/mask path rather than the model export itself. Merely
  changing the sentinel to `-10000` or replacing the output projection with
  convolution did not fix it.
- [Rewritten denoising graph](../results/smolvla_rk3588_denoise_addmask.json):
  the constant Boolean mask is encoded as additive `0/-10000` logits. ONNX
  still matches PyTorch (MAE `4.85e-7`); the RK3588 full 16-layer output
  has cosine `0.999987`, MAE `0.0044` versus original PyTorch. Twenty
  warmed NPU single-step runs: median `43.0 ms`, p95 `43.6 ms`.
- [Full-path matched-input run](../results/smolvla_rk3588_npu_vision_denoise_hybrid.json):
  CPU `39.660 s`, NPU vision plus CPU denoising `36.641 s`, and NPU vision
  plus NPU denoising `7.899 s`. Ten CPU steps took `29.255 s`; ten NPU
  steps plus suffix embedding and cache transfer took `0.520 s`. Final
  action cosine versus CPU is `0.999798`, MAE `0.0119`. Holding NPU vision
  fixed, NPU versus CPU denoising produces action cosine `0.999991`, MAE
  `0.00289`.

The vision graph accepts normalized `[1,3,512,512]` images after LeRobot's
resize/pad preprocessing, not raw camera pixels. The matched-input result is
one deterministic smoke case, not a success-rate or equivalence test. The
language prefix remains CPU-bound, and prompt lengths other than the exported
70-token prefix are unsupported by the denoising RKNN graph. General policy
quality requires evaluation on real task episodes.
