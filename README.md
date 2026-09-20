# Fast-LeWM on RK3588

Fast-LeWM PushT planning on RK3588 with a paper-aligned terminal-only CEM rollout and heterogeneous CPU/NPU execution.

## Current Result

![Aligned CEM latency breakdown](breakdown.png)

Tested on RK3588 with four Cortex-A76 cores pinned, three-core NPU, RKNN Runtime 2.3.2, driver 0.9.8, and the official pretrained PushT checkpoint.

| Metric | CPU | CPU + NPU FP16 | Result |
|---|---:|---:|---:|
| Complete CEM solve | 6.39 s | 3.75 s | **1.71x speedup** |
| Image encoding | 267 ms | 292 ms | CPU in both runs |
| Action-prefix encoder | 2.70 s | 2.87 s | CPU in both runs |
| Terminal predictor + projection | 3.39 s | 555 ms | **6.10x contribution speedup** |

Workload: `300` candidates, `30` CEM iterations, `top-k=30`, five action blocks, one terminal latent per candidate. Raw measurements are in [results/latest_benchmark.json](results/latest_benchmark.json).

The isolated terminal predictor benchmark is:

| Backend | Batch 300 latency |
|---|---:|
| CPU | 118.70 ms |
| NPU FP16 | 17.05 ms |
| Speedup | **6.96x** |

FP16 output agreement against PyTorch: cosine similarity `0.999993`, MAE `0.00150`, maximum absolute error `0.01130`.

## Main Conclusion

The earlier conclusion that RK3588 NPU was no faster than CPU was caused by an implementation mismatch. The old graph predicted all five future latent tokens and discarded the first four when computing the CEM terminal cost. It also instantiated pretrained weights with incorrect head groupings.

The corrected planning path is:

```text
five action blocks
  -> official action-prefix encoder (6 heads x 32)
  -> terminal prefix token [B, 1, 192]
  -> official predictor (16 value heads x 64)
  -> pred_proj
  -> terminal latent cost
```

With this alignment, the fused NPU terminal predictor is about seven times faster than CPU and reduces a complete `300 x 30` CEM solve from 6.39 seconds to 3.75 seconds. NPU acceleration is therefore effective, but the CPU action-prefix encoder is now the dominant bottleneck at roughly 77% of heterogeneous solve time. The current configuration is still not suitable for high-frequency closed-loop control without further planning-budget or action-encoder optimization.

The paper reports `8.0 s` dynamics time and `28.3 s` full CEM time on an NVIDIA RTX 4090. Those absolute numbers are not directly comparable with this single-environment RK3588 run. This repository reports complete-solve and per-module timing explicitly to avoid mixing one CEM iteration with a full 30-iteration solve.

## Correctness Fixes

- Terminal-only predictor input and output are fixed at `[300, 1, 192]`.
- `pred_proj` is fused into the RKNN graph.
- Action-prefix configuration is restored to `6 x 32`.
- Predictor configuration is restored to `16 x 64`.
- Expanded latents are made contiguous before the action encoder.
- CPU and NPU paths use the same terminal cost semantics.
- NPU handles in `board_eval_v2.py` are correctly shared with `get_cost()`.
- Planner command-line CEM iteration and sample settings now take effect.

## Repository Layout

```text
Fast-LeWorldModel/                  Official model source and deployment artifacts
  config/                           Official training/evaluation configuration
  weights/                          Official pretrained PushT checkpoint
  onnx_out/                         Terminal predictor ONNX
  predictor_terminal_*.rknn         Terminal predictor RKNN FP16
  vit_encoder.rknn                  ViT RKNN model
  export_terminal_predictor.py      Checkpoint -> aligned terminal ONNX
  convert_to_rknn.py                ONNX -> RKNN conversion
results/latest_benchmark.json       Latest raw measurements
scripts/plot_breakdown.py           Breakdown figure generator
rk3588_planner_server.py            Board-side planner service
board_eval_v2.py                    Board-side official CEM evaluation path
run_pusht_eval_official.py          Host-side evaluation client
breakdown.png                       Latest complete-CEM breakdown
```

## Reproduction

Export the fused terminal predictor:

```bash
conda run -n stable-wm python Fast-LeWorldModel/export_terminal_predictor.py \
  --checkpoint Fast-LeWorldModel/weights/Fast-lewm_pusht_object.ckpt \
  --outdir Fast-LeWorldModel/onnx_out \
  --batch 300
```

Convert it on the RKNN Toolkit2 Linux environment:

```bash
python Fast-LeWorldModel/convert_to_rknn.py \
  --onnx Fast-LeWorldModel/onnx_out/predictor_terminal_with_proj_b300.onnx \
  --out Fast-LeWorldModel/predictor_terminal_with_proj_b300_fp16.rknn \
  --dtype fp16
```

Deploy and start the board-side planner:

```bash
scp -F ~/.ssh/config_rknn \
  rk3588_planner_server.py \
  Fast-LeWorldModel/predictor_terminal_with_proj_b300_fp16.rknn \
  rk3588:/root/Fast-LeWorldModel/

ssh -F ~/.ssh/config_rknn rk3588 \
  'cd /root/Fast-LeWorldModel && taskset -c 4-7 \
   /root/miniconda3/envs/fast-lewm/bin/python -u rk3588_planner_server.py \
   --mode npu --cem-steps 30 --num-samples 300'
```

Regenerate the chart:

```bash
python scripts/plot_breakdown.py
```

## Remaining Work

1. Run the corrected implementation over the full episode set and report success rate with confidence intervals.
2. Optimize or replace the CPU action-prefix encoder, now the dominant latency component.
3. Build INT8 only with real latent/action-prefix calibration data and revalidate candidate ranking and task success.
