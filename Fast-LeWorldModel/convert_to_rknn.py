"""
Fast-LeWM ONNX → RKNN 转换脚本。

将导出的 ONNX 模型转换为 RKNN 格式，用于 RK3588 NPU 推理。
支持 FP16（无需校准数据）和 INT8（用随机数据校准，仅用于流程测试）。

用法：
  python convert_to_rknn.py --onnx ./onnx_out/predictor_S300.onnx --out ./rknn_out/predictor_S300_fp16.rknn --dtype fp16
  python convert_to_rknn.py --onnx ./onnx_out/action_encoder_S300.onnx --out ./rknn_out/action_encoder_S300_fp16.rknn --dtype fp16
"""
import argparse, os, sys, time
import numpy as np

from rknn.api import RKNN


def _get_input_specs(onnx_path):
    """读取 ONNX 输入名称、静态 shape 和 RKNN 通道数。"""
    import onnx
    model = onnx.load(onnx_path)
    specs = []
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"RKNN export requires static input shapes: {inp.name}={shape}")
        # NCHW 或 NCD 格式：通道数是第二个维度
        ch = shape[1] if len(shape) >= 2 and shape[1] > 0 else 1
        specs.append((inp.name, shape, ch))
    return specs


def convert_onnx_to_rknn(onnx_path, rknn_path, dtype="fp16", calib_samples=20, verbose=True):
    """将 ONNX 转换为 RKNN。"""
    assert dtype in ("fp16", "int8"), "dtype 必须是 fp16 或 int8"
    rknn = RKNN(verbose=verbose)

    # 动态读取输入通道数，设置 mean/std（非图像输入用全0/全1）
    input_specs = _get_input_specs(onnx_path)
    channels = [spec[2] for spec in input_specs]
    mean_values = [[0.0] * ch for ch in channels]
    std_values = [[1.0] * ch for ch in channels]
    print(f"  输入通道数: {channels}, mean_values: {mean_values}")

    # 1. 配置
    print(f"[1/4] 配置 RKNN (dtype={dtype})...")
    ret = rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform="rk3588",
        quant_img_RGB2BGR=False,
        optimization_level=0,  # 完全禁用优化避免 fold_constant bug
    )
    if ret != 0:
        print(f"  config 失败: {ret}")
        return False

    # 2. 加载 ONNX
    print(f"[2/4] 加载 ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  load_onnx 失败: {ret}")
        return False

    # 3. 构建（量化）
    print(f"[3/4] 构建 RKNN (dtype={dtype})...")
    t0 = time.time()
    if dtype == "fp16":
        ret = rknn.build(do_quantization=False)
    else:
        # INT8：默认随机数据只用于验证转换流程。正式精度评测
        # 必须改用真实 CEM latent/action-prefix 数据。
        calib_dir = "/tmp/rknn_calib"
        os.makedirs(calib_dir, exist_ok=True)
        dataset_rows = []
        for i in range(calib_samples):
            paths = []
            for input_name, shape, _ in input_specs:
                path = os.path.join(calib_dir, f"{input_name}_{i}.npy")
                np.save(path, np.random.randn(*shape).astype(np.float32))
                paths.append(path)
            dataset_rows.append(" ".join(paths))
        dataset = os.path.join(calib_dir, "dataset.txt")
        with open(dataset, "w") as f:
            f.write("\n".join(dataset_rows))
            f.write("\n")
        ret = rknn.build(do_quantization=True, dataset=dataset)
    build_time = time.time() - t0
    if ret != 0:
        print(f"  build 失败: {ret}")
        return False
    print(f"  build 耗时: {build_time:.1f}s")

    # 4. 导出
    print(f"[4/4] 导出 RKNN: {rknn_path}")
    os.makedirs(os.path.dirname(rknn_path), exist_ok=True)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  export_rknn 失败: {ret}")
        return False

    size_mb = os.path.getsize(rknn_path) / 1024 / 1024
    print(f"  导出成功: {rknn_path} ({size_mb:.2f} MB)")
    rknn.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    ap.add_argument("--out", required=True, help="输出 RKNN 路径")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    ok = convert_onnx_to_rknn(args.onnx, args.out, args.dtype, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
