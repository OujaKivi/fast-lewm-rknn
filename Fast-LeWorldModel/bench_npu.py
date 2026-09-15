"""
Fast-LeWM NPU 推理性能测试脚本。

在 RK3588 NPU 上加载 RKNN 模型，测量推理延迟、吞吐量、NPU 利用率。

用法：
  python bench_npu.py --rknn ./action_encoder_S300_fp16.rknn --name action_encoder --batch 300 --iters 100
  python bench_npu.py --rknn ./predictor_S300_fp16.rknn --name predictor --batch 300 --iters 100
"""
import argparse, os, sys, time, json
import numpy as np

from rknnlite.api import RKNNLite


def bench_npu(rknn_path, name, batch=300, iters=100, warmup=10):
    """在 NPU 上测试推理性能。"""
    print(f"\n{'='*60}")
    print(f"NPU 性能测试: {name}")
    print(f"模型: {rknn_path}")
    print(f"batch={batch}, iters={iters}, warmup={warmup}")
    print(f"{'='*60}")

    # 1. 加载 RKNN 模型
    print("\n[1/4] 加载 RKNN 模型...")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print(f"  加载失败: {ret}")
        return None
    print(f"  加载成功")

    # 2. 初始化 NPU（自动选择核心）
    print("\n[2/4] 初始化 NPU...")
    ret = rknn.init_runtime(core_mask=RKNNLite.RKNN_NPU_CORE_AUTO)
    if ret != 0:
        print(f"  初始化失败: {ret}")
        rknn.release()
        return None
    print(f"  初始化成功 (CORE_AUTO)")

    # 3. 准备输入数据
    print("\n[3/4] 准备输入数据...")
    # 获取模型输入信息
    # action_encoder: actions [batch,5,2], latent [batch,1,192]
    # predictor: latent [batch,1,192], act_emb [batch,1,192]
    # 用随机数据测试
    if "action_encoder" in name or "encoder" in name:
        inputs = [
            np.random.randn(batch, 5, 2).astype(np.float32),
            np.random.randn(batch, 1, 192).astype(np.float32),
        ]
        input_names = ["actions", "latent"]
    else:
        inputs = [
            np.random.randn(batch, 1, 192).astype(np.float32),
            np.random.randn(batch, 1, 192).astype(np.float32),
        ]
        input_names = ["latent", "act_emb"]

    for i, (inp, n) in enumerate(zip(inputs, input_names)):
        print(f"  输入 {i}: {n} shape={inp.shape}, dtype={inp.dtype}")

    # 4. 推理性能测试
    print(f"\n[4/4] 推理测试 (warmup={warmup}, iters={iters})...")

    # warmup
    for _ in range(warmup):
        outputs = rknn.inference(inputs=inputs)

    # 正式测试
    latencies = []
    npu_loads = []
    npu_freq_path = "/sys/class/devfreq/fdab0000.npu/cur_freq"
    npu_load_path = "/sys/class/devfreq/fdab0000.npu/load"

    for i in range(iters):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        # 读取 NPU 负载（每 10 次读一次）
        if i % 10 == 0:
            try:
                with open(npu_load_path) as f:
                    load = int(f.read().strip())
                npu_loads.append(load)
            except:
                pass

    # 统计
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    throughput = batch / (mean_lat / 1000)  # candidates/sec

    # NPU 频率
    try:
        with open(npu_freq_path) as f:
            npu_freq = int(f.read().strip()) / 1e6  # MHz
    except:
        npu_freq = -1

    avg_npu_load = np.mean(npu_loads) if npu_loads else -1

    # 输出
    print(f"\n{'='*60}")
    print(f"结果: {name}")
    print(f"{'='*60}")
    print(f"  平均延迟:   {mean_lat:.3f} ms ± {std_lat:.3f}")
    print(f"  P50 延迟:   {p50:.3f} ms")
    print(f"  P95 延迟:   {p95:.3f} ms")
    print(f"  P99 延迟:   {p99:.3f} ms")
    print(f"  最小延迟:   {min_lat:.3f} ms")
    print(f"  最大延迟:   {max_lat:.3f} ms")
    print(f"  吞吐量:     {throughput:.1f} candidates/sec")
    print(f"  单候选延迟: {mean_lat/batch:.4f} ms/candidate")
    print(f"  NPU 频率:   {npu_freq:.0f} MHz")
    print(f"  NPU 利用率: {avg_npu_load:.1f}% (采样 {len(npu_loads)} 次)")
    print(f"  输出 shape: {[o.shape for o in outputs]}")

    # 释放
    rknn.release()

    result = {
        "name": name,
        "rknn_path": rknn_path,
        "batch": batch,
        "iters": iters,
        "mean_latency_ms": float(mean_lat),
        "std_latency_ms": float(std_lat),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "min_ms": float(min_lat),
        "max_ms": float(max_lat),
        "throughput_cand_per_sec": float(throughput),
        "latency_per_cand_ms": float(mean_lat / batch),
        "npu_freq_mhz": float(npu_freq),
        "npu_util_pct": float(avg_npu_load),
        "output_shapes": [list(o.shape) for o in outputs],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True, help="RKNN 模型路径")
    ap.add_argument("--name", default="model", help="模型名称")
    ap.add_argument("--batch", type=int, default=300, help="batch size")
    ap.add_argument("--iters", type=int, default=100, help="测试迭代次数")
    ap.add_argument("--warmup", type=int, default=10, help="预热次数")
    ap.add_argument("--out", default=None, help="结果输出 JSON 路径")
    args = ap.parse_args()

    if not os.path.exists(args.rknn):
        print(f"RKNN 文件不存在: {args.rknn}")
        sys.exit(1)

    result = bench_npu(args.rknn, args.name, args.batch, args.iters, args.warmup)

    if result and args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
