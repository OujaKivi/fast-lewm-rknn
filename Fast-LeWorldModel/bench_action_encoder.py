"""
action_encoder CPU vs NPU FP16 vs NPU INT8 效率对比
锁频后运行，对比推理延迟、吞吐量
"""
import time
import numpy as np
import os
import sys

sys.path.insert(0, '/root/Fast-LeWorldModel')

# ============ 配置 ============
BATCH = 300
WARMUP = 10
REPEAT = 50
ACTIONS_SHAPE = (BATCH, 5, 2)
LATENT_SHAPE = (BATCH, 1, 192)

def bench_cpu():
    """CPU 侧：torch ActionPrefixEmbedder"""
    import torch
    
    print("=== CPU 侧 (torch ActionPrefixEmbedder, 4×A76) ===")
    
    # 绑大核
    os.sched_setaffinity(0, {4, 5, 6, 7})
    
    # 用项目里的模型构建函数
    from bench_fast_lewm import build_models
    act_enc, _ = build_models('cpu')
    act_enc.eval()
    
    params = sum(p.numel() for p in act_enc.parameters())
    print(f"参数量: {params/1e6:.2f}M")
    
    actions = torch.randn(*ACTIONS_SHAPE)
    latent = torch.randn(*LATENT_SHAPE)
    
    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = act_enc(actions, latent=latent)
    
    # benchmark
    times = []
    with torch.no_grad():
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            out = act_enc(actions, latent=latent)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    print(f"输出 shape: {out.shape}")
    print(f"延迟: mean={times.mean():.2f}ms, median={np.median(times):.2f}ms, p99={np.percentile(times,99):.2f}ms")
    print(f"吞吐量: {BATCH / (times.mean()/1000):.1f} candidates/s")
    print(f"单候选延迟: {times.mean()/BATCH:.3f}ms/candidate")
    return times.mean()

def bench_npu(model_path, label, input_dtype=np.float32):
    """NPU 侧：RKNN 模型"""
    from rknnlite.api import RKNNLite
    
    print(f"\n=== NPU 侧 ({label}) ===")
    
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"加载 RKNN 失败: {ret}")
        return None
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print(f"初始化 runtime 失败: {ret}")
        return None
    
    actions = np.random.randn(*ACTIONS_SHAPE).astype(input_dtype)
    latent = np.random.randn(*LATENT_SHAPE).astype(input_dtype)
    
    # warmup
    for _ in range(WARMUP):
        _ = rknn.inference(inputs=[actions, latent])
    
    # benchmark
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[actions, latent])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    print(f"输出 shape: {out[0].shape}, dtype: {out[0].dtype}")
    print(f"延迟: mean={times.mean():.2f}ms, median={np.median(times):.2f}ms, p99={np.percentile(times,99):.2f}ms")
    print(f"吞吐量: {BATCH / (times.mean()/1000):.1f} candidates/s")
    print(f"单候选延迟: {times.mean()/BATCH:.3f}ms/candidate")
    
    rknn.release()
    return times.mean()

if __name__ == '__main__':
    print("=" * 70)
    print("action_encoder CPU vs NPU FP16 vs NPU INT8 效率对比")
    print(f"batch={BATCH}, warmup={WARMUP}, repeat={REPEAT}, 锁最高频")
    print("=" * 70)
    
    cpu_ms = bench_cpu()
    fp16_ms = bench_npu('/root/Fast-LeWorldModel/action_encoder_S300_fp16.rknn', 'RKNN FP16', np.float32)
    int8_ms = bench_npu('/root/Fast-LeWorldModel/action_encoder_S300_int8.rknn', 'RKNN INT8', np.int8)
    
    print("\n" + "=" * 70)
    print("总结对比")
    print("=" * 70)
    print(f"{'指标':<22} {'CPU':<15} {'NPU FP16':<15} {'NPU INT8':<15}")
    print("-" * 67)
    print(f"{'batch延迟(ms)':<22} {cpu_ms:<15.2f} {fp16_ms:<15.2f} {int8_ms:<15.2f}")
    print(f"{'单候选延迟(ms)':<22} {cpu_ms/BATCH:<15.3f} {fp16_ms/BATCH:<15.3f} {int8_ms/BATCH:<15.3f}")
    print(f"{'吞吐量(cand/s)':<22} {BATCH/(cpu_ms/1000):<15.1f} {BATCH/(fp16_ms/1000):<15.1f} {BATCH/(int8_ms/1000):<15.1f}")
    print(f"{'vs CPU':<22} {'1.00x':<15} {cpu_ms/fp16_ms:<15.2f}x {cpu_ms/int8_ms:<15.2f}x")
    print(f"{'vs FP16':<22} {'-':<15} {'1.00x':<15} {fp16_ms/int8_ms:<15.2f}x")
    
    print("\n结论:")
    if int8_ms < cpu_ms and int8_ms < fp16_ms:
        print(f"  INT8 最快！比 CPU 快 {(cpu_ms/int8_ms-1)*100:.0f}%，比 FP16 快 {(fp16_ms/int8_ms-1)*100:.0f}%")
        print(f"  action_encoder 适合用 INT8 量化上 NPU")
    elif int8_ms < cpu_ms:
        print(f"  INT8 比 CPU 快 {(cpu_ms/int8_ms-1)*100:.0f}%，但比 FP16 慢 {(int8_ms/fp16_ms-1)*100:.0f}%")
    else:
        print(f"  INT8 比 CPU 慢 {(int8_ms/cpu_ms-1)*100:.0f}%，INT8 量化未带来收益")
        print(f"  可能原因：小模型 NPU 启动开销占比高，INT8 量化/反量化开销抵消了计算加速")
