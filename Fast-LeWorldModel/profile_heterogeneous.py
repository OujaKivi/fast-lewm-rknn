"""
Fast-LeWM 异构调度 profiling（方案 A：CPU action_encoder + NPU predictor）。

在 profile_fast_lewm.py 基础上，把 rollout 中的 predictor 替换为 NPU 推理，
细粒度测量：
  - npu_transfer_in:  torch→numpy + 输入准备
  - npu_inference:    RKNN inference 调用
  - npu_transfer_out: numpy→torch + 输出解析
  - npu_total:        以上合计（每步 predictor）

对比模式：
  --mode cpu : 全 CPU（基线，与 profile_fast_lewm.py 一致）
  --mode npu : action_encoder CPU + predictor NPU（方案 A）

用法：
  python profile_heterogeneous.py --mode npu --tag het_npu --e1
  python profile_heterogeneous.py --mode cpu --tag het_cpu --e1
  python profile_heterogeneous.py --mode both --tag het_compare --e1
"""
import argparse, json, os, sys, time, statistics, resource
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from module import ActionPrefixEmbedder, ARPredictor

# ---- 超参（与 profile_fast_lewm.py 一致）----
EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
CEM_SAMPLES, CEM_ITERS, CEM_TOPK = 300, 30, 30

# 默认 RKNN 模型路径（板端）
DEFAULT_RKNN_PATH = "/root/Fast-LeWorldModel/predictor_S300_fp16_v4.rknn"
DEFAULT_ENC_RKNN_PATH = "/root/Fast-LeWorldModel/action_encoder_S300_fp16.rknn"


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def now_ns():
    return time.perf_counter_ns()


def stats_ms(ts_ns):
    if not ts_ns:
        return {}
    ms = [t / 1e6 for t in ts_ns]
    return {
        "mean_ms": round(statistics.mean(ms), 4),
        "median_ms": round(statistics.median(ms), 4),
        "min_ms": round(min(ms), 4),
        "p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 4),
        "count": len(ms),
    }


def build_models(device):
    act_enc = ActionPrefixEmbedder(
        input_dim=ACTION_DIM, smoothed_dim=32, emb_dim=EMBED_DIM, mlp_scale=4,
        temporal_mixer_type="transformer", use_positional_encoding=True,
        transformer_depth=AP_DEPTH, transformer_heads=AP_HEADS,
        transformer_dim_head=AP_DIMHEAD, transformer_mlp_dim=AP_MLP,
        use_latent_condition=True, latent_dim=EMBED_DIM,
    ).to(device).eval()
    predictor = ARPredictor(
        depth=PRED_DEPTH, mlp_dim=PRED_MLP, input_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
        value_heads=PRED_VHEADS, value_dim_head=PRED_VDIMHEAD, heads=PRED_VHEADS,
        dim_head=PRED_VDIMHEAD, action_fusion_hidden_dim=PRED_FUSION,
        action_fusion_zero_init=True, token_processing="batch",
    ).to(device).eval()
    return act_enc, predictor


def build_npu_predictor(rknn_path):
    """构建 NPU 版 predictor（方案 A）。"""
    from npu_predictor import NPUPredictor
    return NPUPredictor(rknn_path)


def build_vit(device):
    try:
        import timm
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).to(device).eval()
    except Exception:
        return None


# ============================================================
# 异构版 CEM 决策：action_encoder CPU + predictor NPU/CPU
# ============================================================
@torch.no_grad()
def cem_decision_het(act_enc, predictor, vit, S, iters, topk, steps, device,
                     use_npu=False, npu_predictor=None, use_npu_enc=False, npu_encoder=None):
    """
    异构版 CEM 决策。
    use_npu=True 时，predictor 用 NPU（npu_predictor）。
    use_npu_enc=True 时，action_encoder 用 NPU（npu_encoder）。
    组合：
      use_npu=False, use_npu_enc=False → CPU 全量基线
      use_npu=True,  use_npu_enc=False → 方案A异构（CPU enc + NPU pred）
      use_npu=True,  use_npu_enc=True  → NPU 全量基线（NPU enc + NPU pred）

    细粒度计时：
      vi_enc, cem_sample, tensor_prep, act_enc_cpu / act_enc_npu, rollout_total,
      npu_transfer_in, npu_inference, npu_transfer_out,
      score, elite_select
    """
    T = defaultdict(list)
    t_total0 = now_ns()

    # ---- 视觉编码 ----
    if vit is not None:
        t0 = now_ns()
        img = torch.randn(2, 3, 224, 224, device=device)
        feats = vit(img)
        z_cur = feats[0:1].unsqueeze(1)
        z_goal = feats[1:2].unsqueeze(1)
        T["vi_enc"].append(now_ns() - t0)
    else:
        z_cur = torch.randn(1, 1, EMBED_DIM, device=device)
        z_goal = torch.randn(1, 1, EMBED_DIM, device=device)

    mu = torch.zeros(1, steps, ACTION_DIM, device=device)
    sigma = torch.ones(1, steps, ACTION_DIM, device=device) * 0.5

    pred_fn = npu_predictor if use_npu else predictor
    enc_fn = npu_encoder if use_npu_enc else act_enc
    enc_tag = "act_enc_npu" if use_npu_enc else "act_enc_cpu"

    for i in range(iters):
        # 1) CEM 采样
        t0 = now_ns()
        noise = torch.randn(S, steps, ACTION_DIM, device=device)
        actions = mu + sigma * noise
        T["cem_sample"].append(now_ns() - t0)

        # 2) tensor prep
        t0 = now_ns()
        latent = z_cur.expand(S, -1, -1).contiguous()
        act_input = actions.contiguous()
        T["tensor_prep"].append(now_ns() - t0)

        # 3) action_encoder（CPU 或 NPU）
        t0 = now_ns()
        emb = latent
        act_emb = enc_fn(act_input, return_last_only=True, latent=emb[:, -1:])
        T[enc_tag].append(now_ns() - t0)

        # 4) rollout：5 步 predictor（NPU 或 CPU）
        t_rollout0 = now_ns()
        for _ in range(steps):
            if use_npu:
                # NPU 细粒度计时
                t_tin = now_ns()
                latent_np = emb[:, -1:].detach().cpu().numpy().astype(np.float32, copy=False)
                act_emb_np = act_emb.detach().cpu().numpy().astype(np.float32, copy=False)
                T["npu_transfer_in"].append(now_ns() - t_tin)

                t_inf = now_ns()
                outputs = npu_predictor.rknn.inference(inputs=[latent_np, act_emb_np])
                T["npu_inference"].append(now_ns() - t_inf)

                t_tout = now_ns()
                pred_np = outputs[0]
                pred = torch.from_numpy(pred_np).to(device)
                emb = torch.cat([emb, pred], dim=1)
                T["npu_transfer_out"].append(now_ns() - t_tout)
            else:
                # CPU predictor
                pred = predictor(emb[:, -1:], act_emb)
                emb = torch.cat([emb, pred], dim=1)
        T["rollout_total"].append(now_ns() - t_rollout0)

        # 5) score
        t0 = now_ns()
        final = emb[:, -1:, :]
        cost = ((final - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1)
        T["score"].append(now_ns() - t0)

        # 6) elite selection
        t0 = now_ns()
        _, top_idx = torch.topk(cost, topk, largest=False)
        elite = actions[top_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 0.01
        T["elite_select"].append(now_ns() - t0)

    total_ns = now_ns() - t_total0
    return T, mu, total_ns


def run_e1_het(act_enc, predictor, vit, device, use_npu=False, npu_predictor=None,
               use_npu_enc=False, npu_encoder=None,
               repeats=3, iters=CEM_ITERS, S=CEM_SAMPLES):
    """跑 E1 异构版，返回各阶段统计。"""
    # warmup
    cem_decision_het(act_enc, predictor, vit, S, 2, CEM_TOPK, ACTION_BLOCKS,
                     device, use_npu=use_npu, npu_predictor=npu_predictor,
                     use_npu_enc=use_npu_enc, npu_encoder=npu_encoder)

    all_T = defaultdict(list)
    totals = []
    for r in range(repeats):
        T, mu, total = cem_decision_het(
            act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS,
            device, use_npu=use_npu, npu_predictor=npu_predictor,
            use_npu_enc=use_npu_enc, npu_encoder=npu_encoder)
        for k, v in T.items():
            all_T[k].extend(v)
        totals.append(total)

    stage_stats = {k: stats_ms(v) for k, v in all_T.items()}
    stage_total_ms = {k: round(v["median_ms"] * iters, 3) for k, v in stage_stats.items()}
    total_median_ms = round(statistics.median(totals) / 1e6, 3)

    # NPU 每步平均（5步/iter × iters）
    npu_per_step = {}
    if use_npu:
        for key in ["npu_transfer_in", "npu_inference", "npu_transfer_out"]:
            if key in stage_stats:
                npu_per_step[key] = stage_stats[key]

    # 模式标签
    if use_npu and use_npu_enc:
        mode_tag = "npu_full"
    elif use_npu:
        mode_tag = "npu_het"
    else:
        mode_tag = "cpu"

    return {
        "mode": mode_tag,
        "iters": iters, "samples": S, "repeats": repeats,
        "stage_per_iter_ms": stage_stats,
        "stage_total_ms": stage_total_ms,
        "total_decision_median_ms": total_median_ms,
        "total_decision_s": round(total_median_ms / 1000, 3),
        "npu_per_step_ms": npu_per_step,
    }


# ============================================================
# 输出一致性验证：CPU vs NPU predictor 输出对比
# ============================================================
@torch.no_grad()
def verify_output_consistency(act_enc, cpu_pred, npu_pred, device, S=CEM_SAMPLES, steps=ACTION_BLOCKS):
    """验证 NPU predictor 与 CPU predictor 输出的一致性（cosine similarity / MSE）。"""
    latent = torch.randn(S, 1, EMBED_DIM, device=device)
    act_input = torch.randn(S, steps, ACTION_DIM, device=device)
    act_emb = act_enc(act_input, return_last_only=True, latent=latent)

    # CPU 单步
    cpu_pred_out = cpu_pred(latent, act_emb)
    # NPU 单步
    npu_pred_out = npu_pred(latent, act_emb)

    cpu_np = cpu_pred_out.cpu().numpy()
    npu_np = npu_pred_out.cpu().numpy()

    # cosine similarity（逐样本）
    cos_sim = np.sum(cpu_np * npu_np, axis=-1) / (
        np.linalg.norm(cpu_np, axis=-1) * np.linalg.norm(npu_np, axis=-1) + 1e-8)
    # MSE
    mse = np.mean((cpu_np - npu_np) ** 2)
    # 相对误差
    rel_err = np.mean(np.abs(cpu_np - npu_np)) / (np.mean(np.abs(cpu_np)) + 1e-8)

    return {
        "cos_sim_mean": round(float(cos_sim.mean()), 6),
        "cos_sim_min": round(float(cos_sim.min()), 6),
        "mse": round(float(mse), 8),
        "relative_l1_error": round(float(rel_err), 6),
        "cpu_output_mean": round(float(cpu_np.mean()), 6),
        "npu_output_mean": round(float(npu_np.mean()), 6),
        "note": "随机权重下数值对比无实际意义，仅验证推理流程正确性；加载预训练权重后需重新验证",
    }


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "npu", "npu_full", "both", "all"], default="all",
                    help="cpu=全CPU基线, npu=CPU enc+NPU pred(方案A), npu_full=NPU enc+NPU pred, both=cpu+npu, all=三者对比")
    ap.add_argument("--rknn", default=DEFAULT_RKNN_PATH, help="RKNN predictor 模型路径")
    ap.add_argument("--rknn-enc", default=DEFAULT_ENC_RKNN_PATH, help="RKNN action_encoder 模型路径")
    ap.add_argument("--tag", default="het_profile")
    ap.add_argument("--e1", action="store_true", default=True)
    ap.add_argument("--verify", action="store_true", help="验证 CPU/NPU 输出一致性")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=CEM_ITERS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    threads = torch.get_num_threads()

    print(f"[build] models on {device}, threads={threads} ...", flush=True)
    act_enc, cpu_predictor = build_models(device)
    vit = build_vit(device)

    npu_predictor = None
    npu_encoder = None
    need_npu_pred = args.mode in ["npu", "both", "all", "npu_full"]
    need_npu_enc = args.mode in ["npu_full", "all"]

    if need_npu_pred:
        print(f"[build] NPU predictor from {args.rknn} ...", flush=True)
        npu_predictor = build_npu_predictor(args.rknn)
        print("  NPU predictor 加载成功", flush=True)

    if need_npu_enc:
        print(f"[build] NPU action_encoder from {args.rknn_enc} ...", flush=True)
        from npu_predictor import NPUActionEncoder
        npu_encoder = NPUActionEncoder(args.rknn_enc)
        print("  NPU action_encoder 加载成功", flush=True)

    rep = 2 if args.quick else args.repeats
    iters = 5 if args.quick else args.iters

    res = {
        "tag": args.tag,
        "mode": args.mode,
        "torch": torch.__version__,
        "platform": os.uname().machine,
        "torch_threads": threads,
        "rknn_predictor": args.rknn if npu_predictor else None,
        "rknn_encoder": args.rknn_enc if npu_encoder else None,
        "config": {"embed_dim": EMBED_DIM, "action_blocks": ACTION_BLOCKS,
                   "cem_samples": CEM_SAMPLES, "cem_iters": iters, "cem_topk": CEM_TOPK},
    }

    # 输出一致性验证
    if args.verify and npu_predictor is not None:
        print("[verify] CPU vs NPU predictor 输出一致性 ...", flush=True)
        res["verify_consistency_pred"] = verify_output_consistency(
            act_enc, cpu_predictor, npu_predictor, device)

    # E1
    if args.e1:
        run_cpu = args.mode in ["cpu", "both", "all"]
        run_npu_het = args.mode in ["npu", "both", "all"]
        run_npu_full = args.mode in ["npu_full", "all"]

        if run_cpu:
            print(f"[E1][CPU] 端到端分解, iters={iters}, repeats={rep} ...", flush=True)
            res["E1_cpu"] = run_e1_het(
                act_enc, cpu_predictor, vit, device, use_npu=False,
                repeats=rep, iters=iters)

        if run_npu_het:
            print(f"[E1][NPU异构(方案A)] 端到端分解, iters={iters}, repeats={rep} ...", flush=True)
            res["E1_npu_het"] = run_e1_het(
                act_enc, cpu_predictor, vit, device, use_npu=True, npu_predictor=npu_predictor,
                use_npu_enc=False, repeats=rep, iters=iters)

        if run_npu_full:
            print(f"[E1][NPU全量] 端到端分解, iters={iters}, repeats={rep} ...", flush=True)
            res["E1_npu_full"] = run_e1_het(
                act_enc, cpu_predictor, vit, device, use_npu=True, npu_predictor=npu_predictor,
                use_npu_enc=True, npu_encoder=npu_encoder, repeats=rep, iters=iters)

        # 加速比汇总
        speedup = {}
        if "E1_cpu" in res:
            speedup["cpu_total_s"] = res["E1_cpu"]["total_decision_s"]
        if "E1_npu_het" in res:
            speedup["npu_het_total_s"] = res["E1_npu_het"]["total_decision_s"]
            if "E1_cpu" in res:
                speedup["het_vs_cpu"] = round(
                    res["E1_cpu"]["total_decision_s"] / res["E1_npu_het"]["total_decision_s"], 3)
        if "E1_npu_full" in res:
            speedup["npu_full_total_s"] = res["E1_npu_full"]["total_decision_s"]
            if "E1_cpu" in res:
                speedup["full_vs_cpu"] = round(
                    res["E1_cpu"]["total_decision_s"] / res["E1_npu_full"]["total_decision_s"], 3)
            if "E1_npu_het" in res:
                speedup["het_vs_full"] = round(
                    res["E1_npu_full"]["total_decision_s"] / res["E1_npu_het"]["total_decision_s"], 3)
        if speedup:
            res["speedup"] = speedup
            print(f"\n[结果汇总]", flush=True)
            for k, v in speedup.items():
                print(f"  {k}: {v}", flush=True)

    res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    out = f"/root/Fast-LeWorldModel/profile_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("SAVED", out, flush=True)

    if npu_predictor:
        npu_predictor.release()
    if npu_encoder:
        npu_encoder.release()


if __name__ == "__main__":
    main()
