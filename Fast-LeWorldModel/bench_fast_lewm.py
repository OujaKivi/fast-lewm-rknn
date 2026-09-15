"""
Fast-LeWorldModel edge efficiency benchmark (CPU, aarch64/RK3588).
直接复用仓库 module.py 中的真实网络类，按 config/train/Fast-lewm.yaml 维度构建，
并按 eval CEM 规划器(num_samples=300, action_num_blocks=5)的真实批量规模测推理效率。
不依赖数据集 / checkpoint / mujoco / stable_worldmodel，仅需 torch + einops(+可选 timm)。
"""
import argparse, json, os, sys, time, statistics, resource

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from module import ActionPrefixEmbedder, ARPredictor

# ---- 来自 config/train/Fast-lewm.yaml 的真实超参 ----
EMBED_DIM = 192
ACTION_DIM = 2                 # PushT-v1 单步动作维度(x,y)
ACTION_BLOCKS = 5              # plan_config.action_num_blocks
# action_prefix
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
# predictor
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
# CEM
CEM_SAMPLES, CEM_ITERS = 300, 30


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def timed(fn, warmup=3, repeat=15):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {
        "mean_ms": round(statistics.mean(ts), 3),
        "median_ms": round(statistics.median(ts), 3),
        "min_ms": round(min(ts), 3),
        "p95_ms": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 3),
        "repeat": repeat,
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


def build_vit_tiny(device):
    """近似视觉编码器 ViT-Tiny(224): embed192/depth12/heads3。patch14 非 timm 标准,用 patch16 量级参考。"""
    try:
        import timm
        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).to(device).eval()
        return m, "timm vit_tiny_patch16_224 (近似; 仓库为 patch14 ViT-tiny)"
    except Exception as e:
        return None, f"timm 不可用,跳过视觉编码器: {e}"


@torch.no_grad()
def rollout(act_enc, predictor, S, steps, device):
    """对应 jepa.JEPA.rollout 的世界模型推演: 逐步 action_encoder + predictor。"""
    latent = torch.randn(S, 1, EMBED_DIM, device=device)
    emb = latent
    for _ in range(steps):
        act_seq = torch.randn(S, ACTION_BLOCKS, ACTION_DIM, device=device)
        act_emb = act_enc(act_seq, return_last_only=True, latent=emb[:, -1:])
        pred = predictor(emb[:, -1:], act_emb)
        emb = torch.cat([emb, pred], dim=1)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--samples-scan", default="1,32,100,300")
    args = ap.parse_args()

    device = "cpu"
    ncpu = os.cpu_count()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    act_enc, predictor = build_models(device)
    vit, vit_note = build_vit_tiny(device)

    res = {
        "tag": args.tag,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": os.uname().machine,
        "os_cpus": ncpu,
        "torch_threads": threads,
        "affinity": sorted(os.sched_getaffinity(0)),
        "params": {
            "action_encoder": n_params(act_enc),
            "predictor": n_params(predictor),
            "wm_total": n_params(act_enc) + n_params(predictor),
            "vit_tiny_approx": n_params(vit) if vit is not None else None,
        },
        "config": {"embed_dim": EMBED_DIM, "action_blocks": ACTION_BLOCKS,
                   "cem_samples": CEM_SAMPLES, "cem_iters": CEM_ITERS},
        "vit_note": vit_note,
        "units": {},
    }

    scan = [int(x) for x in args.samples_scan.split(",")]
    rep = 8 if args.quick else 15

    # 1) 动作编码器 / 预测器 随候选数 S 扫描
    for S in scan:
        act_seq = torch.randn(S, ACTION_BLOCKS, ACTION_DIM)
        lat = torch.randn(S, 1, EMBED_DIM)
        c = torch.randn(S, 1, EMBED_DIM)
        res["units"][f"action_encoder_S{S}"] = timed(
            lambda: act_enc(act_seq, return_last_only=True, latent=lat), repeat=rep)
        res["units"][f"predictor_S{S}"] = timed(
            lambda: predictor(lat, c), repeat=rep)

    # 2) 完整世界模型 rollout(5步) —— 一次 get_cost 的核心
    for S in scan:
        res["units"][f"rollout5_S{S}"] = timed(
            lambda: rollout(act_enc, predictor, S, ACTION_BLOCKS, device), repeat=max(5, rep // 2))

    # 3) 视觉 ViT 编码(224x224, 当前帧+目标各一次)
    if vit is not None:
        x = torch.randn(1, 3, 224, 224)
        res["units"]["vit_encode_1x224"] = timed(lambda: vit(x), warmup=2, repeat=max(5, rep // 2))

    # 4) 由分项估算单次 CEM 规划决策(get_action)耗时
    r300 = res["units"]["rollout5_S300"]["median_ms"]
    vit_ms = res["units"].get("vit_encode_1x224", {}).get("median_ms", 0.0)
    cem_ms = CEM_ITERS * r300 + 2 * vit_ms
    res["estimated_single_mpc_decision"] = {
        "cem_iters": CEM_ITERS,
        "rollout5_S300_median_ms": r300,
        "vit_encode_x2_ms": round(2 * vit_ms, 3),
        "total_ms": round(cem_ms, 2),
        "total_s": round(cem_ms / 1000.0, 3),
        "note": "30次迭代 x 300候选rollout + 当前帧/目标视觉编码; 未含CEM采样/Python开销与mujoco渲染",
    }
    res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    out = f"/root/Fast-LeWorldModel/bench_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("SAVED", out)


if __name__ == "__main__":
    main()
