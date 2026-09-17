"""
predictor FP16 vs INT8 对比测试：
1. 单步推理延迟对比
2. 输出误差（量化损失）
3. 端到端 CEM 规划时间对比
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from module import ActionPrefixEmbedder, ARPredictor
from rknnlite.api import RKNNLite

EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
CEM_SAMPLES, CEM_ITERS, CEM_TOPK = 300, 30, 30

def now_ns():
    return time.perf_counter_ns()

def build_act_enc(device):
    return ActionPrefixEmbedder(
        input_dim=ACTION_DIM, smoothed_dim=32, emb_dim=EMBED_DIM, mlp_scale=4,
        temporal_mixer_type="transformer", use_positional_encoding=True,
        transformer_depth=AP_DEPTH, transformer_heads=AP_HEADS,
        transformer_dim_head=AP_DIMHEAD, transformer_mlp_dim=AP_MLP,
        use_latent_condition=True, latent_dim=EMBED_DIM,
    ).to(device).eval()

def load_rknn(path):
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    return rknn

def npu_rollout(rknn, emb, act_emb, steps, device):
    """NPU rollout，自动处理 int8 输入输出。"""
    for _ in range(steps):
        latent_np = emb[:, -1:].detach().cpu().numpy().astype(np.float32, copy=False)
        act_emb_np = act_emb.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = rknn.inference(inputs=[latent_np, act_emb_np])
        pred = torch.from_numpy(outputs[0]).to(device)
        emb = torch.cat([emb, pred], dim=1)
    return emb

@torch.no_grad()
def cem_plan(act_enc, rknn, S, iters, topk, steps, device):
    T = defaultdict(list)
    t0_total = now_ns()
    z_cur = torch.randn(1, 1, EMBED_DIM, device=device)
    z_goal = torch.randn(1, 1, EMBED_DIM, device=device)
    mu = torch.zeros(1, steps, ACTION_DIM, device=device)
    sigma = torch.ones(1, steps, ACTION_DIM, device=device) * 0.5
    for i in range(iters):
        noise = torch.randn(S, steps, ACTION_DIM, device=device)
        actions = mu + sigma * noise
        t0 = now_ns()
        latent = z_cur.expand(S, -1, -1).contiguous()
        act_emb = act_enc(actions, return_last_only=True, latent=latent[:, -1:])
        T["enc_cpu"].append(now_ns() - t0)
        t0 = now_ns()
        emb = latent
        emb = npu_rollout(rknn, emb, act_emb, steps, device)
        T["pred_npu"].append(now_ns() - t0)
        final = emb[:, -1:, :]
        cost = ((final - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1)
        _, top_idx = torch.topk(cost, topk, largest=False)
        elite = actions[top_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 0.01
    total_ns = now_ns() - t0_total
    return T, total_ns

def test_accuracy(rknn_fp16, rknn_int8, device, num_tests=10):
    """测试 INT8 量化误差。"""
    print(f"\n=== 量化精度测试（{num_tests} 组随机输入）===")
    errors = []
    rel_errors = []
    for i in range(num_tests):
        latent = torch.randn(CEM_SAMPLES, 1, EMBED_DIM, device=device)
        act_emb = torch.randn(CEM_SAMPLES, 1, EMBED_DIM, device=device)
        latent_np = latent.cpu().numpy().astype(np.float32)
        act_emb_np = act_emb.cpu().numpy().astype(np.float32)
        out_fp16 = rknn_fp16.inference(inputs=[latent_np, act_emb_np])[0]
        out_int8 = rknn_int8.inference(inputs=[latent_np, act_emb_np])[0]
        diff = np.abs(out_fp16 - out_int8)
        mae = np.mean(diff)
        max_err = np.max(diff)
        rel = np.mean(diff / (np.abs(out_fp16) + 1e-8))
        errors.append(mae)
        rel_errors.append(rel)
        if i < 3:
            print(f"  测试 {i}: MAE={mae:.6f}, MaxErr={max_err:.6f}, RelErr={rel*100:.3f}%")
    avg_mae = np.mean(errors)
    avg_rel = np.mean(rel_errors)
    print(f"  平均 MAE: {avg_mae:.6f}")
    print(f"  平均相对误差: {avg_rel*100:.3f}%")
    print(f"  FP16 输出范围: [{out_fp16.min():.4f}, {out_fp16.max():.4f}], std={out_fp16.std():.4f}")
    return {"avg_mae": float(avg_mae), "avg_rel_err": float(avg_rel)}

def test_latency(rknn_fp16, rknn_int8, device, warmup=5, repeats=20):
    """测试单步推理延迟。"""
    print(f"\n=== 单步推理延迟测试（warmup={warmup}, repeats={repeats}）===")
    latent = torch.randn(CEM_SAMPLES, 1, EMBED_DIM, device=device)
    act_emb = torch.randn(CEM_SAMPLES, 1, EMBED_DIM, device=device)
    latent_np = latent.cpu().numpy().astype(np.float32)
    act_emb_np = act_emb.cpu().numpy().astype(np.float32)
    # warmup
    for _ in range(warmup):
        rknn_fp16.inference(inputs=[latent_np, act_emb_np])
        rknn_int8.inference(inputs=[latent_np, act_emb_np])
    # test
    fp16_times = []
    int8_times = []
    for _ in range(repeats):
        t0 = now_ns()
        rknn_fp16.inference(inputs=[latent_np, act_emb_np])
        fp16_times.append(now_ns() - t0)
        t0 = now_ns()
        rknn_int8.inference(inputs=[latent_np, act_emb_np])
        int8_times.append(now_ns() - t0)
    fp16_mean = np.mean(fp16_times) / 1e6
    int8_mean = np.mean(int8_times) / 1e6
    fp16_p99 = np.percentile(fp16_times, 99) / 1e6
    int8_p99 = np.percentile(int8_times, 99) / 1e6
    speedup = fp16_mean / int8_mean
    print(f"  FP16: mean={fp16_mean:.2f}ms, p99={fp16_p99:.2f}ms")
    print(f"  INT8: mean={int8_mean:.2f}ms, p99={int8_p99:.2f}ms")
    print(f"  加速比: {speedup:.2f}x")
    return {"fp16_mean_ms": float(fp16_mean), "int8_mean_ms": float(int8_mean),
            "fp16_p99_ms": float(fp16_p99), "int8_p99_ms": float(int8_p99),
            "speedup": float(speedup)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", default="/root/Fast-LeWorldModel/predictor_S300_fp16_v4.rknn")
    ap.add_argument("--int8", default="/root/Fast-LeWorldModel/predictor_S300_int8.rknn")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    device = "cpu"
    os.sched_setaffinity(0, {4, 5, 6, 7})
    print("=== 加载模型 ===")
    act_enc = build_act_enc(device)
    rknn_fp16 = load_rknn(args.fp16)
    rknn_int8 = load_rknn(args.int8)
    print(f"FP16: {args.fp16}")
    print(f"INT8: {args.int8}")
    results = {}
    # 1. 精度测试
    results["accuracy"] = test_accuracy(rknn_fp16, rknn_int8, device)
    # 2. 延迟测试
    results["latency"] = test_latency(rknn_fp16, rknn_int8, device)
    # 3. 端到端 CEM 测试
    print(f"\n=== 端到端 CEM 规划对比（{args.iters} iters × {args.repeats} repeats）===")
    for name, rknn in [("FP16", rknn_fp16), ("INT8", rknn_int8)]:
        totals = []
        for r in range(args.repeats):
            T, total = cem_plan(act_enc, rknn, CEM_SAMPLES, args.iters, CEM_TOPK, ACTION_BLOCKS, device)
            total_s = total / 1e9
            enc_ms = np.mean(T["enc_cpu"]) / 1e6
            pred_ms = np.mean(T["pred_npu"]) / 1e6
            totals.append(total_s)
            print(f"  {name} repeat {r}: total={total_s:.2f}s, enc={enc_ms:.1f}ms/iter, pred={pred_ms:.1f}ms/iter")
        avg = np.mean(totals)
        results[name] = {"avg_total_s": float(avg), "totals": [float(t) for t in totals]}
        print(f"  {name} 平均: {avg:.2f}s")
    if "FP16" in results and "INT8" in results:
        speedup = results["FP16"]["avg_total_s"] / results["INT8"]["avg_total_s"]
        print(f"\n  端到端加速比: {speedup:.2f}x")
        results["e2e_speedup"] = float(speedup)
    with open("/root/Fast-LeWorldModel/pred_int8_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: pred_int8_result.json")
    rknn_fp16.release()
    rknn_int8.release()

if __name__ == "__main__":
    main()
