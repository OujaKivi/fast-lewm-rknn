"""转换 predictor INT8 RKNN（batch=300），生成校准数据。"""
import torch, sys, os, onnx, numpy as np
sys.path.insert(0, "/home/wjw/rknn-work")
from module import ARPredictor

EMBED_DIM = 192
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
B = 300

print("=== 生成校准数据 ===")
os.makedirs("calib_data", exist_ok=True)
calib_list = []
for i in range(20):
    latent = np.random.randn(B, 1, EMBED_DIM).astype(np.float32)
    act_emb = np.random.randn(B, 1, EMBED_DIM).astype(np.float32)
    np.save(f"calib_data/latent_{i}.npy", latent)
    np.save(f"calib_data/act_emb_{i}.npy", act_emb)
    calib_list.append(f"/home/wjw/rknn-work/calib_data/latent_{i}.npy /home/wjw/rknn-work/calib_data/act_emb_{i}.npy")

with open("calib_data/calib_list.txt", "w") as f:
    f.write("\n".join(calib_list))
print(f"生成 {len(calib_list)} 组校准数据")

print("\n=== 转换 INT8 RKNN ===")
from rknn.api import RKNN
rknn = RKNN(verbose=False)
# predictor 输入 [B,1,192]，channel=1
rknn.config(
    mean_values=[[0], [0]],
    std_values=[[1], [1]],
    target_platform="rk3588",
    quantized_dtype="asymmetric_quantized-8",
)
rknn.load_onnx(model="onnx_out/predictor_S300_fixed.onnx")
rknn.build(do_quantization=True, dataset="calib_data/calib_list.txt")
out_path = "rknn_out/predictor_S300_int8.rknn"
rknn.export_rknn(out_path)
print(f"INT8 RKNN: {os.path.getsize(out_path)/1024/1024:.2f} MB")
rknn.release()
print("\n=== 完成 ===")
