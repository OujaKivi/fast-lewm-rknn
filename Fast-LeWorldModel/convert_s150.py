"""导出 batch=150 的 predictor ONNX，修复全零权重，转换 RKNN FP16。"""
import torch, sys, os, onnx, numpy as np
sys.path.insert(0, "/home/wjw/rknn-work")
from module import ARPredictor

EMBED_DIM = 192
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
B = 150

print("=== 导出 ONNX ===")
predictor = ARPredictor(
    depth=PRED_DEPTH, mlp_dim=PRED_MLP, input_dim=EMBED_DIM,
    hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
    value_heads=PRED_VHEADS, value_dim_head=PRED_VDIMHEAD, heads=PRED_VHEADS,
    dim_head=PRED_VDIMHEAD, action_fusion_hidden_dim=PRED_FUSION,
    action_fusion_zero_init=True, token_processing="batch",
).eval()

latent = torch.randn(B, 1, EMBED_DIM)
act_emb = torch.randn(B, 1, EMBED_DIM)
onnx_path = "onnx_out/predictor_S150_fixed.onnx"
torch.onnx.export(predictor, (latent, act_emb), onnx_path,
    opset_version=14, input_names=["latent", "act_emb"], output_names=["pred"])
print(f"ONNX: {os.path.getsize(onnx_path)/1024/1024:.2f} MB")

print("\n=== 修复全零权重 ===")
m = onnx.load(onnx_path)
count = 0
for init in m.graph.initializer:
    if init.dims and init.raw_data:
        arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        if np.all(arr == 0) and arr.size > 0:
            arr[:] = np.random.randn(arr.size).astype(np.float32) * 0.01
            init.raw_data = arr.tobytes()
            count += 1
onnx.save(m, onnx_path)
print(f"修复 {count} 个全零初始值")

print("\n=== 转换 RKNN FP16 ===")
from rknn.api import RKNN
rknn = RKNN(verbose=False)
# predictor 输入 [B,1,192]，rknn 把倒数第二维1当channel
rknn.config(mean_values=[[0], [0]], std_values=[[1], [1]], target_platform="rk3588")
rknn.load_onnx(model=onnx_path)
rknn.build(do_quantization=False)
out_path = "rknn_out/predictor_S150_fp16.rknn"
rknn.export_rknn(out_path)
print(f"RKNN: {os.path.getsize(out_path)/1024/1024:.2f} MB")
rknn.release()
print("\n=== 完成 ===")
