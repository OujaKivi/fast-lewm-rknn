"""
Fast-LeWorldModel 端侧异构 profiling（阶段一：E1/E3/E4/E5）。

在已有 bench_fast_lewm.py 基础上扩展，核心是把一次完整 CEM 规划决策
分解为可独立测量的阶段，并做 Fast-LeWM(action-prefix) 与受控
LeWM(逐步自回归) 对照。

阶段定义（对应计划书 4.1）：
  vi_enc        视觉编码（当前帧+目标，每决策一次）
  cem_sample    CEM 候选动作采样（高斯采样+加噪）
  tensor_prep   candidate reshape / expand / layout 整理
  rollout_model 世界模型推演（action_enc + predictor，5步）
  score         latent cost 计算
  elite_select  top-k elite 选择 + mu/sigma 更新
  python_loop   纯 Python 循环开销（迭代间）

用法：
  python profile_fast_lewm.py --e1 --tag e1_big4        # 端到端分解
  python profile_fast_lewm.py --e3 --tag e3_scan        # batch 扫描
  python profile_fast_lewm.py --e4 --tag e4_runtime      # runtime 微基准
  python profile_fast_lewm.py --all --tag full           # 全部
"""
import argparse, json, os, sys, time, statistics, resource
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from module import ActionPrefixEmbedder, ARPredictor

# ---- 超参（与 bench 一致，来自 config/train/Fast-lewm.yaml）----
EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
CEM_SAMPLES, CEM_ITERS, CEM_TOPK = 300, 30, 30


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def now_ns():
    return time.perf_counter_ns()


def stats_ms(ts_ns):
    """由 ns 列表生成统计(ms)。"""
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


def build_vit(device):
    try:
        import timm
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).to(device).eval()
    except Exception:
        return None


# ============================================================
# E1: 端到端 CEM 循环 latency 分解
# ============================================================
@torch.no_grad()
def cem_decision(act_enc, predictor, vit, S, iters, topk, steps, device, mode="fast"):
    """
    模拟一次完整 CEM 规划决策。mode:
      fast : Fast-LeWM —— action-prefix encoder 一次编码完整动作序列，多步复用。
      ar   : 受控 LeWM 对照 —— 每步重新编码单步动作，dynamics 调用 steps 次。
    返回 (各阶段 ns 列表 dict, 最终 mu, 总耗时 ns)。
    """
    T = defaultdict(list)
    t_total0 = now_ns()

    # ---- 视觉编码（每决策一次：当前帧 + 目标）----
    if vit is not None:
        t0 = now_ns()
        img = torch.randn(2, 3, 224, 224, device=device)
        feats = vit(img)  # (2, 192)
        z_cur = feats[0:1].unsqueeze(1)   # (1,1,192)
        z_goal = feats[1:2].unsqueeze(1)
        T["vi_enc"].append(now_ns() - t0)
    else:
        z_cur = torch.randn(1, 1, EMBED_DIM, device=device)
        z_goal = torch.randn(1, 1, EMBED_DIM, device=device)

    mu = torch.zeros(1, steps, ACTION_DIM, device=device)
    sigma = torch.ones(1, steps, ACTION_DIM, device=device) * 0.5

    for i in range(iters):
        # 1) CEM 采样
        t0 = now_ns()
        noise = torch.randn(S, steps, ACTION_DIM, device=device)
        actions = mu + sigma * noise  # (S, steps, 2)
        T["cem_sample"].append(now_ns() - t0)

        # 2) tensor prep（expand / reshape / 整理成模型输入）
        t0 = now_ns()
        latent = z_cur.expand(S, -1, -1).contiguous()  # (S,1,192)
        act_input = actions.contiguous()                  # (S,steps,2)
        T["tensor_prep"].append(now_ns() - t0)

        # 3) rollout 模型计算
        t0 = now_ns()
        emb = latent
        if mode == "fast":
            # action-prefix 一次编码完整序列，多步 predictor 复用
            act_emb = act_enc(act_input, return_last_only=True, latent=emb[:, -1:])
            for _ in range(steps):
                pred = predictor(emb[:, -1:], act_emb)
                emb = torch.cat([emb, pred], dim=1)
        else:  # ar / LeWM 对照：每步重新编码单步动作
            for s in range(steps):
                step_act = act_input[:, s:s+1, :]
                act_emb = act_enc(step_act, return_last_only=True, latent=emb[:, -1:])
                pred = predictor(emb[:, -1:], act_emb)
                emb = torch.cat([emb, pred], dim=1)
        T["rollout_model"].append(now_ns() - t0)

        # 4) score（最终 latent 与 goal 的 L2 距离）
        t0 = now_ns()
        final = emb[:, -1:, :]  # (S,1,192)
        cost = ((final - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1)  # (S,)
        T["score"].append(now_ns() - t0)

        # 5) elite selection（top-k + 更新 mu/sigma）
        t0 = now_ns()
        _, top_idx = torch.topk(cost, topk, largest=False)
        elite = actions[top_idx]  # (topk, steps, 2)
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 0.01
        T["elite_select"].append(now_ns() - t0)

    total_ns = now_ns() - t_total0
    return T, mu, total_ns


def run_e1(act_enc, predictor, vit, device, repeats=3, iters=CEM_ITERS, S=CEM_SAMPLES):
    """E1: Fast vs AR 对照，各跑 repeats 次完整决策。"""
    out = {"fast": {}, "ar": {}}
    for mode in ["fast", "ar"]:
        # warmup 1 次（短 iters）
        cem_decision(act_enc, predictor, vit, S, 2, CEM_TOPK, ACTION_BLOCKS, device, mode)
        all_T = defaultdict(list)
        totals = []
        for r in range(repeats):
            T, mu, total = cem_decision(act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS, device, mode)
            for k, v in T.items():
                all_T[k].extend(v)
            totals.append(total)
        # 每阶段统计（按 iter 聚合后的 median）
        stage_stats = {k: stats_ms(v) for k, v in all_T.items()}
        # 每阶段总耗时（median per iter * iters）
        stage_total_ms = {k: round(v["median_ms"] * iters, 3) for k, v in stage_stats.items()}
        total_median_ms = round(statistics.median(totals) / 1e6, 3)
        # 占比
        sum_stage = sum(stage_total_ms.values())
        pct = {k: round(v / sum_stage * 100, 2) if sum_stage > 0 else 0 for k, v in stage_total_ms.items()}
        out[mode] = {
            "iters": iters, "samples": S, "repeats": repeats,
            "stage_per_iter_ms": stage_stats,
            "stage_total_ms": stage_total_ms,
            "stage_pct": pct,
            "total_decision_median_ms": total_median_ms,
            "total_decision_s": round(total_median_ms / 1000, 3),
        }
    return out


# ============================================================
# E3: candidate batch 扫描
# ============================================================
@torch.no_grad()
def rollout_only(act_enc, predictor, S, steps, device, mode="fast"):
    """纯 rollout（不含 CEM 循环），用于 batch 扫描。"""
    latent = torch.randn(S, 1, EMBED_DIM, device=device)
    act_input = torch.randn(S, steps, ACTION_DIM, device=device)
    emb = latent
    if mode == "fast":
        act_emb = act_enc(act_input, return_last_only=True, latent=emb[:, -1:])
        for _ in range(steps):
            pred = predictor(emb[:, -1:], act_emb)
            emb = torch.cat([emb, pred], dim=1)
    else:
        for s in range(steps):
            act_emb = act_enc(act_input[:, s:s+1, :], return_last_only=True, latent=emb[:, -1:])
            pred = predictor(emb[:, -1:], act_emb)
            emb = torch.cat([emb, pred], dim=1)
    return emb


def run_e3(act_enc, predictor, device, scan="16,32,64,128,256,512", repeats=5):
    """E3: 扫描 candidate batch size。"""
    Ss = [int(x) for x in scan.split(",")]
    out = {}
    # warmup
    rollout_only(act_enc, predictor, 64, ACTION_BLOCKS, device, "fast")
    for S in Ss:
        ts_fast, ts_ar = [], []
        for _ in range(repeats):
            t0 = now_ns(); rollout_only(act_enc, predictor, S, ACTION_BLOCKS, device, "fast"); ts_fast.append(now_ns() - t0)
            t0 = now_ns(); rollout_only(act_enc, predictor, S, ACTION_BLOCKS, device, "ar"); ts_ar.append(now_ns() - t0)
        f_med = statistics.median(ts_fast) / 1e6
        a_med = statistics.median(ts_ar) / 1e6
        out[str(S)] = {
            "fast_rollout5_median_ms": round(f_med, 3),
            "ar_rollout5_median_ms": round(a_med, 3),
            "fast_latency_per_candidate_ms": round(f_med / S, 5),
            "fast_throughput_candidates_per_s": round(S / (f_med / 1000), 2),
            "speedup_fast_vs_ar": round(a_med / f_med, 3) if f_med > 0 else None,
        }
    return out


# ============================================================
# E4: CPU 侧 runtime 微基准
# ============================================================
def run_e4(device, S=CEM_SAMPLES, repeats=50):
    """E4: 隔离测量非模型阶段的绝对耗时。"""
    out = {}
    steps = ACTION_BLOCKS
    actions = torch.randn(S, steps, ACTION_DIM, device=device)
    z_cur = torch.randn(1, 1, EMBED_DIM, device=device)
    z_goal = torch.randn(1, 1, EMBED_DIM, device=device)
    emb = torch.randn(S, steps + 1, EMBED_DIM, device=device)
    cost = torch.randn(S, device=device)

    def bench(name, fn, rep=repeats):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(rep):
            t0 = now_ns(); fn(); ts.append(now_ns() - t0)
        out[name] = stats_ms(ts)

    # cem_sample: 高斯采样
    bench("cem_sample_gaussian_S300", lambda: torch.randn(S, steps, ACTION_DIM, device=device) * 0.5)
    # tensor_prep: expand + contiguous
    bench("tensor_prep_expand_contiguous", lambda: (z_cur.expand(S, -1, -1).contiguous(), actions.contiguous()))
    # latent cat（rollout 内每步的 cat，模拟 5 次）
    def multi_cat():
        e = emb[:, :1, :]
        for s in range(steps):
            e = torch.cat([e, emb[:, s+1:s+2, :]], dim=1)
        return e
    bench("latent_cat_5steps", multi_cat)
    # score: L2 cost
    bench("score_l2_cost", lambda: ((emb[:, -1:, :] - z_goal.expand(S, -1, -1)) ** 2).sum(dim=-1).squeeze(-1))
    # elite_select: topk + mean + std
    def elite():
        _, idx = torch.topk(cost, CEM_TOPK, largest=False)
        el = actions[idx]
        return el.mean(dim=0, keepdim=True), el.std(dim=0, keepdim=True)
    bench("elite_select_topk30_mean_std", elite)
    # python 循环空转（30 次）
    def py_loop():
        acc = 0
        for i in range(CEM_ITERS):
            acc += i
        return acc
    bench("python_loop_30iters_empty", py_loop, rep=200)
    # 参考：一次 randn(S,1,192)（作为 tensor 分配基线）
    bench("tensor_alloc_randn_S192", lambda: torch.randn(S, 1, EMBED_DIM, device=device))
    return out


# ============================================================
# E5: 精度基线 + 资源记录
# ============================================================
@torch.no_grad()
def run_e5(act_enc, predictor, device, S=64, steps=ACTION_BLOCKS):
    """记录多步 rollout 的 latent 数值统计（作为 FP32 精度基线）+ 内存。"""
    latent = torch.randn(S, 1, EMBED_DIM, device=device)
    act_input = torch.randn(S, steps, ACTION_DIM, device=device)
    act_emb = act_enc(act_input, return_last_only=True, latent=latent)
    emb = latent
    for _ in range(steps):
        pred = predictor(emb[:, -1:], act_emb)
        emb = torch.cat([emb, pred], dim=1)
    final = emb[:, -1, :]  # (S,192)
    rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    npu_freq = None
    try:
        with open("/sys/class/devfreq/fdab0000.npu/cur_freq") as f:
            npu_freq = int(f.read().strip())
    except Exception:
        pass
    return {
        "final_latent_mean": round(float(final.mean()), 6),
        "final_latent_std": round(float(final.std()), 6),
        "final_latent_abs_max": round(float(final.abs().max()), 6),
        "final_latent_norm_mean": round(float(final.norm(dim=-1).mean()), 6),
        "max_rss_mb": rss_mb,
        "npu_cur_freq_hz": npu_freq,
        "note": "FP32 CPU 推理精度基线；NPU 量化后应与此对比多步 rollout 误差",
    }


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="profile")
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e3", action="store_true")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--e5", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--e1-repeats", type=int, default=3)
    ap.add_argument("--e1-iters", type=int, default=CEM_ITERS)
    ap.add_argument("--e3-scan", default="16,32,64,128,256,512")
    ap.add_argument("--quick", action="store_true", help="快速冒烟：减少重复")
    args = ap.parse_args()

    if args.all:
        args.e1 = args.e3 = args.e4 = args.e5 = True
    if not (args.e1 or args.e3 or args.e4 or args.e5):
        args.e1 = True  # 默认至少 E1

    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    print(f"[build] models on {device}, threads={threads} ...", flush=True)
    act_enc, predictor = build_models(device)
    vit = build_vit(device)

    res = {
        "tag": args.tag,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": os.uname().machine,
        "os_cpus": os.cpu_count(),
        "torch_threads": threads,
        "affinity": sorted(os.sched_getaffinity(0)),
        "params": {
            "action_encoder": n_params(act_enc),
            "predictor": n_params(predictor),
            "wm_total": n_params(act_enc) + n_params(predictor),
            "vit_tiny_approx": n_params(vit) if vit is not None else None,
        },
        "config": {"embed_dim": EMBED_DIM, "action_blocks": ACTION_BLOCKS,
                   "cem_samples": CEM_SAMPLES, "cem_iters": CEM_ITERS, "cem_topk": CEM_TOPK},
    }

    if args.e1:
        rep = 2 if args.quick else args.e1_repeats
        iters = 5 if args.quick else args.e1_iters
        print(f"[E1] end-to-end CEM decomposition, iters={iters}, repeats={rep} ...", flush=True)
        res["E1"] = run_e1(act_enc, predictor, vit, device, repeats=rep, iters=iters)

    if args.e3:
        rep = 2 if args.quick else 5
        print(f"[E3] candidate batch scan {args.e3_scan} ...", flush=True)
        res["E3"] = run_e3(act_enc, predictor, device, scan=args.e3_scan, repeats=rep)

    if args.e4:
        print("[E4] runtime micro-benchmarks ...", flush=True)
        res["E4"] = run_e4(device)

    if args.e5:
        print("[E5] precision baseline + resources ...", flush=True)
        res["E5"] = run_e5(act_enc, predictor, device)

    res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    out = f"/root/Fast-LeWorldModel/profile_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
