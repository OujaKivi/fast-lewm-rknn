"""
vi_enc (ViT-tiny) CPU vs NPU 效率对比测试
锁频后运行，对比推理延迟、吞吐量
"""
import time
import numpy as np
import sys

# ============ 配置 ============
BATCH = 2          # 和 ONNX 导出时一致
WARMUP = 20
REPEAT = 100
INPUT_SHAPE = (BATCH, 3, 224, 224)

def bench_cpu():
    """CPU 侧：timm vit_tiny_patch16_224"""
    import torch
    import timm
    
    print("=== CPU 侧 (timm vit_tiny_patch16_224) ===")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    model.eval()
    
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params/1e6:.2f}M")
    
    dummy = torch.randn(*INPUT_SHAPE)
    
    # 绑大核
    import os
    os.sched_setaffinity(0, {4, 5, 6, 7})
    
    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)
    
    # benchmark
    times = []
    with torch.no_grad():
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            out = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    print(f"输出 shape: {out.shape}")
    print(f"延迟: mean={times.mean():.2f}ms, median={np.median(times):.2f}ms, p99={np.percentile(times,99):.2f}ms")
    print(f"吞吐量: {BATCH / (times.mean()/1000):.1f} samples/s")
    print(f"单次候选延迟: {times.mean()/BATCH:.2f}ms/candidate")
    return times.mean()

def bench_npu():
    """NPU 侧：RKNN 模型"""
    from rknnlite.api import RKNNLite
    
    print("\n=== NPU 侧 (RKNN FP16) ===")
    rknn = RKNNLite()
    ret = rknn.load_rknn('/root/Fast-LeWorldModel/vit_tiny_b2_fp16.rknn')
    if ret != 0:
        print("加载 RKNN 失败")
        return None
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print("初始化 runtime 失败")
        return None
    
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    
    # warmup
    for _ in range(WARMUP):
        _ = rknn.inference(inputs=[dummy])
    
    # benchmark
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[dummy])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    print(f"输出 shape: {out[0].shape}")
    print(f"延迟: mean={times.mean():.2f}ms, median={np.median(times):.2f}ms, p99={np.percentile(times,99):.2f}ms")
    print(f"吞吐量: {BATCH / (times.mean()/1000):.1f} samples/s")
    print(f"单次候选延迟: {times.mean()/BATCH:.2f}ms/candidate")
    
    rknn.release()
    return times.mean()

if __name__ == '__main__':
    print("=" * 60)
    print("vi_enc (ViT-tiny) CPU vs NPU 效率对比")
    print(f"batch={BATCH}, warmup={WARMUP}, repeat={REPEAT}")
    print("=" * 60)
    
    cpu_ms = bench_cpu()
    npu_ms = bench_npu()
    
    print("\n" + "=" * 60)
    print("总结对比")
    print("=" * 60)
    print(f"{'指标':<20} {'CPU (4×A76)':<18} {'NPU (3核)':<18} {'NPU/CPU':<10}")
    print("-" * 66)
    print(f"{'batch延迟(ms)':<20} {cpu_ms:<18.2f} {npu_ms:<18.2f} {npu_ms/cpu_ms:<10.2f}x")
    print(f"{'单候选延迟(ms)':<20} {cpu_ms/BATCH:<18.2f} {npu_ms/BATCH:<18.2f} {npu_ms/cpu_ms:<10.2f}x")
    print(f"{'吞吐量(samples/s)':<20} {BATCH/(cpu_ms/1000):<18.1f} {BATCH/(npu_ms/1000):<18.1f} {cpu_ms/npu_ms:<10.2f}x")
    
    if npu_ms < cpu_ms:
        print(f"\n结论: NPU 比 CPU 快 {(cpu_ms/npu_ms - 1)*100:.1f}%，vi_enc 适合上 NPU")
    else:
        print(f"\n结论: NPU 比 CPU 慢 {(npu_ms/cpu_ms - 1)*100:.1f}%，vi_enc 不适合上 NPU（小模型 NPU 启动开销占比高）")
