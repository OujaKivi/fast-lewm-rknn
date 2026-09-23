# SmolVLA RK3588 NPU pilot

This pilot moves only the image encoder and connector to the RK3588 NPU. The
language backbone, action expert, and ten denoising steps stay in PyTorch on
four Cortex-A76 cores. The checkpoint is the official `lerobot/smolvla_base`,
with LeRobot 0.4.4, RKNN Toolkit2/Lite2 2.3.2, and NPU driver 0.9.8.

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

## Measurements

- [Vision graph: 20 warmed runs](../results/smolvla_rk3588_vision_rknn.json):
  median 872 ms, p95 877 ms, output cosine 0.999825, MAE 0.0547.
- [Full-path matched-input pilot](../results/smolvla_rk3588_npu_vision_hybrid.json):
  CPU 38.099 s; NPU-vision hybrid 36.281 s. Ten CPU denoising steps account
  for 28.935 s of the CPU path. Final action cosine 0.999788, MAE 0.0109
  with identical initial noise.

The vision graph accepts normalized `[1,3,512,512]` images after LeRobot's
resize/pad preprocessing, not raw camera pixels. The matched-input result is
one deterministic smoke case, not a success-rate or equivalence test. A
full-NPU or useful split-inference claim requires exporting and validating
the language/action path and evaluating real task episodes.
