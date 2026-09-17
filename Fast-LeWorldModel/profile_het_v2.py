"""
方案 B 双缓冲流水线 CEM 规划原型。
同一 iteration 内将 candidate 分两组（各150）：
  CPU 编码 A 组 -> 启动 NPU 推理 A 组（线程，用 S150 模型）
  主线程同时 CPU 编码 B 组 -> 等待 NPU A 完成 -> NPU 推理 B 组
对比方案 A（串行：CPU 编码全部300 -> NPU 推理全部300，用 S300 模型）。
"""
import argparse, json, os, sys, time, threading
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from module import ActionPrefixEmbedder, ARPredictor

EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
CEM_SAMPLES, CEM_ITERS, CEM_TOPK = 300, 30, 30

def now_ns():
    return time.perf_counter_ns()

def build_models(device):
    act_enc = ActionPrefixEmbedder(
        input_dim=ACTION_DIM, smoothed_dim=32, emb_dim=EMBED_DIM, mlp_scale=4,
        temporal_mixer_type="transformer", use_positional_encoding=True,
        transformer_depth=AP_DEPTH, transformer_heads=AP_HEADS,
        transformer_dim_head=AP_DIMHEAD, transformer_mlp_dim=AP_MLP,
        use_latent_condition=True, latent_dim=EMBED_DIM,
    ).to(device).eval()
    return act_enc

def npu_rollout(npu_pred, emb, act_emb, steps, device):
    for _ in range(steps):
        latent_np = emb[:, -1:].detach().cpu().numpy().astype(np.float32, copy=False)
        act_emb_np = act_emb.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = npu_pred.rknn.inference(inputs=[latent_np, act_emb_np])
        pred = torch.from_numpy(outputs[0]).to(device)
        emb = torch.cat([emb, pred], dim=1)
    return emb

@torch.no_grad()
def cem_plan_a(act_enc, npu_pred_s300, S, iters, topk, steps, device):
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
        emb = npu_rollout(npu_pred_s300, emb, act_emb, steps, device)
        T["pred_npu"].append(now_ns() - t0)
        final = emb[:, -1:, :]
        cost = ((final - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1)
        _, top_idx = torch.topk(cost, topk, largest=False)
        elite = actions[top_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 0.01
    total_ns = now_ns() - t0_total
    return T, total_ns

@torch.no_grad()
def cem_plan_b(act_enc, npu_pred_s150, S, iters, topk, steps, device, num_chunks=2):
    T = defaultdict(list)
    t0_total = now_ns()
    chunk_size = S // num_chunks
    z_cur = torch.randn(1, 1, EMBED_DIM, device=device)
    z_goal = torch.randn(1, 1, EMBED_DIM, device=device)
    mu = torch.zeros(1, steps, ACTION_DIM, device=device)
    sigma = torch.ones(1, steps, ACTION_DIM, device=device) * 0.5
    for i in range(iters):
        noise = torch.randn(S, steps, ACTION_DIM, device=device)
        actions = mu + sigma * noise
        latent = z_cur.expand(S, -1, -1).contiguous()
        # 第一组 CPU 编码
        t_enc_start = now_ns()
        c0_actions = actions[:chunk_size].contiguous()
        c0_latent = latent[:chunk_size].contiguous()
        c0_act_emb = act_enc(c0_actions, return_last_only=True, latent=c0_latent[:, -1:])
        T["enc_cpu"].append(now_ns() - t_enc_start)
        # 启动线程 NPU 推理第一组
        npu_result = {}
        def npu_worker():
            emb = c0_latent
            emb = npu_rollout(npu_pred_s150, emb, c0_act_emb, steps, device)
            npu_result["emb"] = emb
        t_npu_start = now_ns()
        npu_thread = threading.Thread(target=npu_worker)
        npu_thread.start()
        # 主线程同时 CPU 编码第二组
        t_enc2_start = now_ns()
        c1_actions = actions[chunk_size:].contiguous()
        c1_latent = latent[chunk_size:].contiguous()
        c1_act_emb = act_enc(c1_actions, return_last_only=True, latent=c1_latent[:, -1:])
        T["enc_cpu_overlap"].append(now_ns() - t_enc2_start)
        # 等待 NPU 第一组完成
        npu_thread.join()
        T["pred_npu_chunk0"].append(now_ns() - t_npu_start)
        c0_emb = npu_result["emb"]
        # NPU 推理第二组
        t_pred2_start = now_ns()
        c1_emb = c1_latent
        c1_emb = npu_rollout(npu_pred_s150, c1_emb, c1_act_emb, steps, device)
        T["pred_npu_chunk1"].append(now_ns() - t_pred2_start)
        # 合并
        emb = torch.cat([c0_emb, c1_emb], dim=0)
        final = emb[:, -1:, :]
        cost = ((final - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1)
        _, top_idx = torch.topk(cost, topk, largest=False)
        elite = actions[top_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 0.01
    total_ns = now_ns() - t0_total
    return T, total_ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a", "b", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rknn-s300", default="/root/Fast-LeWorldModel/predictor_S300_fp16_v4.rknn")
    ap.add_argument("--rknn-s150", default="/root/Fast-LeWorldModel/predictor_S150_fp16.rknn")
    args = ap.parse_args()
    device = "cpu"
    os.sched_setaffinity(0, {4, 5, 6, 7})
    print("=== 构建模型 ===")
    act_enc = build_models(device)
    from npu_predictor import NPUPredictor
    npu_s300 = NPUPredictor(args.rknn_s300)
    npu_s150 = NPUPredictor(args.rknn_s150)
    print(f"NPU S300: {args.rknn_s300}")
    print(f"NPU S150: {args.rknn_s150}")
    results = {}
    if args.mode in ("a", "both"):
        print(f"\n=== 方案 A（串行，S300）===")
        for r in range(args.repeats):
            T, total = cem_plan_a(act_enc, npu_s300, CEM_SAMPLES, args.iters, CEM_TOPK, ACTION_BLOCKS, device)
            total_s = total / 1e9
            enc_ms = np.mean(T["enc_cpu"]) / 1e6
            pred_ms = np.mean(T["pred_npu"]) / 1e6
            print(f"  repeat {r}: total={total_s:.2f}s, enc={enc_ms:.1f}ms/iter, pred={pred_ms:.1f}ms/iter")
            if r == args.repeats - 1:
                results["A"] = {"total_s": total_s, "enc_ms": enc_ms, "pred_ms": pred_ms}
    if args.mode in ("b", "both"):
        print(f"\n=== 方案 B（双缓冲，S150×2）===")
        for r in range(args.repeats):
            T, total = cem_plan_b(act_enc, npu_s150, CEM_SAMPLES, args.iters, CEM_TOPK, ACTION_BLOCKS, device)
            total_s = total / 1e9
            enc0_ms = np.mean(T["enc_cpu"]) / 1e6
            enc_overlap_ms = np.mean(T["enc_cpu_overlap"]) / 1e6
            pred0_ms = np.mean(T["pred_npu_chunk0"]) / 1e6
            pred1_ms = np.mean(T["pred_npu_chunk1"]) / 1e6
            print(f"  repeat {r}: total={total_s:.2f}s, enc0={enc0_ms:.1f}ms, enc_overlap={enc_overlap_ms:.1f}ms, pred0={pred0_ms:.1f}ms, pred1={pred1_ms:.1f}ms")
            if r == args.repeats - 1:
                results["B"] = {"total_s": total_s, "enc0_ms": enc0_ms, "enc_overlap_ms": enc_overlap_ms, "pred0_ms": pred0_ms, "pred1_ms": pred1_ms}
    if "A" in results and "B" in results:
        print(f"\n=== 对比总结 ===")
        print(f"  方案 A: {results['A']['total_s']:.2f}s")
        print(f"  方案 B: {results['B']['total_s']:.2f}s")
        speedup = results["A"]["total_s"] / results["B"]["total_s"]
        print(f"  加速比: {speedup:.2f}x ({(speedup-1)*100:.1f}%)")
        results["speedup"] = speedup
    with open("/root/Fast-LeWorldModel/profile_het_v2_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: profile_het_v2_result.json")

if __name__ == "__main__":
    main()
