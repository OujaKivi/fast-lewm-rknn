"""
根因诊断脚本：区分 NPU 全量模式下 CPU 性能下降的原因
  - 内存带宽竞争？
  - 发热降频（DVFS）？
  - runtime 系统开销？

实验设计（每种模式 10 iter，采样 vi_enc 时间 + CPU 频率 + 温度）：
  mode1 cpu_baseline:  纯 CPU（vi_enc + action_encoder + predictor 都 CPU）
  mode2 npu_het:       异构（CPU enc + NPU pred）
  mode3 npu_full:      全量（NPU enc + NPU pred）
  mode4 sleep_enc:     对照（sleep 182ms 代替 NPU action_encoder + NPU pred）
                        如果 vi_enc 变慢程度 ≈ mode2 → 内存带宽竞争
                        如果 vi_enc 变慢程度 ≈ mode3 → 发热/时间占用
"""
import os, sys, time, json, statistics, subprocess
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module import ActionPrefixEmbedder, ARPredictor

# ---- 超参（与 profile_heterogeneous.py 一致）----
EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768
CEM_SAMPLES, CEM_ITERS, CEM_TOPK = 300, 30, 30

# ---------- 硬件监控 ----------
def read_cpu_freq(cpu_id=4):
    """读取指定 CPU 核的当前频率 (kHz)"""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip())
    except Exception:
        return -1

def read_temp():
    """读取所有 thermal zone 温度，返回最高温 (°C)"""
    temps = []
    for i in range(10):
        try:
            with open(f"/sys/class/thermal/thermal_zone{i}/temp") as f:
                t = int(f.read().strip())
                if t > 1000:
                    t = t // 1000  # 有些是毫摄氏度
                temps.append(t)
        except Exception:
            break
    return max(temps) if temps else -1

def read_npu_load():
    """读取 NPU 利用率"""
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except Exception:
        return "N/A"

# ---------- 模型构建 ----------
def build_vit(device):
    import timm
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).to(device).eval()

def build_models(device="cpu"):
    torch.set_num_threads(8)
    vit = build_vit(device)
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
    return vit, act_enc, predictor

def build_npu(rknn_path):
    from npu_predictor import NPUPredictor
    return NPUPredictor(rknn_path)

def build_npu_enc(rknn_path):
    from npu_predictor import NPUActionEncoder
    return NPUActionEncoder(rknn_path)

# ---------- CEM 决策（带硬件监控） ----------
@torch.no_grad()
def cem_decision_diag(act_enc, predictor, vit, S, iters, topk, steps, device,
                       mode="cpu", npu_pred=None, npu_enc=None, sleep_enc_ms=0):
    """
    mode:
      cpu      - 全 CPU
      npu_het  - CPU enc + NPU pred
      npu_full - NPU enc + NPU pred
      sleep_enc- sleep 代替 NPU enc + NPU pred（对照）
    """
    vi_enc_times = []
    cpu_freqs = []
    temps = []

    # 视觉编码
    t0 = time.perf_counter()
    img = torch.randn(2, 3, 224, 224, device=device)
    feats = vit(img)
    z_cur = feats[0:1].unsqueeze(1)
    z_goal = feats[1:2].unsqueeze(1)
    vi_enc_times.append(time.perf_counter() - t0)
    cpu_freqs.append(read_cpu_freq())
    temps.append(read_temp())

    mu = torch.zeros(1, steps, ACTION_DIM, device=device)
    sigma = torch.ones(1, steps, ACTION_DIM, device=device) * 0.5

    for i in range(iters):
        # CEM 采样
        noise = torch.randn(S, steps, ACTION_DIM, device=device)
        actions = mu + sigma * noise

        # tensor prep
        latent = z_cur.expand(S, -1, -1).contiguous()

        # action_encoder
        if mode == "npu_full":
            act_emb = npu_enc(actions, return_last_only=True, latent=latent[:, -1:])
        elif mode == "sleep_enc":
            # 模拟 NPU action_encoder 的时间占用，但不实际访问内存
            time.sleep(sleep_enc_ms / 1000.0)
            act_emb = act_enc(actions, return_last_only=True, latent=latent[:, -1:])
        else:
            act_emb = act_enc(actions, return_last_only=True, latent=latent[:, -1:])

        # rollout
        emb = latent
        for _ in range(steps):
            if mode in ["npu_het", "npu_full"]:
                pred = npu_pred(emb[:, -1:], act_emb)
            elif mode == "sleep_enc":
                time.sleep(22.6 / 1000.0)  # 模拟 NPU predictor 时间
                pred = predictor(emb[:, -1:], act_emb)
            else:
                pred = predictor(emb[:, -1:], act_emb)
            emb = torch.cat([emb, pred], dim=1)

        # score + elite
        diff = emb[:, 1:] - z_goal
        score = (diff ** 2).sum(dim=(1, 2))
        topk_idx = torch.topk(score, topk, largest=False).indices
        elite = actions[topk_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True) + 1e-6

        # 每 iter 结束采样硬件状态
        cpu_freqs.append(read_cpu_freq())
        temps.append(read_temp())

    return {
        "vi_enc_first_ms": vi_enc_times[0] * 1000,
        "cpu_freq_mean_khz": statistics.mean(cpu_freqs),
        "cpu_freq_min_khz": min(cpu_freqs),
        "cpu_freq_max_khz": max(cpu_freqs),
        "temp_mean_c": statistics.mean(temps),
        "temp_max_c": max(temps),
        "temp_min_c": min(temps),
    }

# ---------- 主流程 ----------
def main():
    device = "cpu"
    iters = 10
    S = CEM_SAMPLES

    print("=== 构建模型 ===", flush=True)
    vit, act_enc, predictor = build_models(device)

    npu_pred = build_npu("/root/Fast-LeWorldModel/predictor_S300_fp16_v4.rknn")
    npu_enc = build_npu_enc("/root/Fast-LeWorldModel/action_encoder_S300_fp16.rknn")

    # 先预热
    print("=== 预热 ===", flush=True)
    cem_decision_diag(act_enc, predictor, vit, S, 2, CEM_TOPK, ACTION_BLOCKS, device, mode="cpu")

    results = {}

    # 模式1: CPU 基线
    print("\n=== mode1: CPU 基线 ===", flush=True)
    time.sleep(2)  # 冷却
    results["cpu_baseline"] = cem_decision_diag(
        act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS, device, mode="cpu")

    # 模式2: NPU 异构
    print("=== mode2: NPU 异构（CPU enc + NPU pred）===", flush=True)
    time.sleep(2)
    results["npu_het"] = cem_decision_diag(
        act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS, device,
        mode="npu_het", npu_pred=npu_pred)

    # 模式3: NPU 全量
    print("=== mode3: NPU 全量（NPU enc + NPU pred）===", flush=True)
    time.sleep(2)
    results["npu_full"] = cem_decision_diag(
        act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS, device,
        mode="npu_full", npu_pred=npu_pred, npu_enc=npu_enc)

    # 模式4: sleep 对照（模拟全量时间但不实际用 NPU 推理）
    print("=== mode4: sleep 对照（sleep 代替 NPU 推理）===", flush=True)
    time.sleep(2)
    results["sleep_enc"] = cem_decision_diag(
        act_enc, predictor, vit, S, iters, CEM_TOPK, ACTION_BLOCKS, device,
        mode="sleep_enc", sleep_enc_ms=182)

    # 输出对比
    print("\n" + "="*70, flush=True)
    print("=== 根因诊断结果 ===", flush=True)
    print("="*70, flush=True)

    modes = ["cpu_baseline", "npu_het", "npu_full", "sleep_enc"]
    mode_names = {
        "cpu_baseline": "CPU基线",
        "npu_het": "NPU异构",
        "npu_full": "NPU全量",
        "sleep_enc": "sleep对照"
    }

    print(f"\n{'指标':<25} {'CPU基线':>10} {'NPU异构':>10} {'NPU全量':>10} {'sleep对照':>10}", flush=True)
    print("-"*70, flush=True)

    base_vi = results["cpu_baseline"]["vi_enc_first_ms"]
    for key, label in [
        ("vi_enc_first_ms", "vi_enc首次(ms)"),
        ("cpu_freq_mean_khz", "CPU均频(kHz)"),
        ("cpu_freq_min_khz", "CPU最低频(kHz)"),
        ("temp_mean_c", "均温(°C)"),
        ("temp_max_c", "最高温(°C)"),
    ]:
        row = f"{label:<25}"
        for m in modes:
            v = results[m][key]
            if key == "vi_enc_first_ms":
                row += f" {v:>10.1f}"
            elif "freq" in key:
                row += f" {v:>10.0f}"
            else:
                row += f" {v:>10.1f}"
        print(row, flush=True)

    # vi_enc 变慢比例
    print("\n--- vi_enc 变慢比例（vs CPU基线）---", flush=True)
    for m in ["npu_het", "npu_full", "sleep_enc"]:
        slowdown = results[m]["vi_enc_first_ms"] / base_vi
        print(f"  {mode_names[m]}: {slowdown:.2f}× ({(slowdown-1)*100:.0f}% 变慢)", flush=True)

    # 根因判断
    print("\n--- 根因判断 ---", flush=True)
    het_slow = results["npu_het"]["vi_enc_first_ms"] / base_vi
    full_slow = results["npu_full"]["vi_enc_first_ms"] / base_vi
    sleep_slow = results["sleep_enc"]["vi_enc_first_ms"] / base_vi

    print(f"  异构变慢: {het_slow:.2f}×", flush=True)
    print(f"  全量变慢: {full_slow:.2f}×", flush=True)
    print(f"  sleep对照变慢: {sleep_slow:.2f}×", flush=True)

    if sleep_slow < full_slow * 0.8:
        print("\n  >> 结论: sleep对照变慢显著小于全量 → 主要是【内存带宽竞争】", flush=True)
        print("     (NPU 推理时实际访问 DDR，与 CPU 抢带宽；sleep 不访问内存)", flush=True)
    elif abs(sleep_slow - full_slow) < 0.15:
        print("\n  >> 结论: sleep对照变慢 ≈ 全量 → 主要是【发热降频】或【时间占用】", flush=True)
        print("     (NPU 高负载导致 SoC 升温，DVFS 降低 CPU 频率)", flush=True)
    else:
        print("\n  >> 结论: 混合因素，内存带宽竞争 + 发热降频都有贡献", flush=True)

    # CPU 频率对比
    base_freq = results["cpu_baseline"]["cpu_freq_mean_khz"]
    full_freq = results["npu_full"]["cpu_freq_mean_khz"]
    if full_freq < base_freq * 0.95:
        print(f"\n  >> CPU 频率下降: {base_freq:.0f} → {full_freq:.0f} kHz "
              f"({(1-full_freq/base_freq)*100:.1f}%) → 确认存在降频", flush=True)
    else:
        print(f"\n  >> CPU 频率无明显下降: {base_freq:.0f} → {full_freq:.0f} kHz → 降频不是主因", flush=True)

    # 保存结果
    out = {
        "modes": results,
        "vi_enc_slowdown": {
            "npu_het": round(het_slow, 3),
            "npu_full": round(full_slow, 3),
            "sleep_enc": round(sleep_slow, 3),
        },
        "cpu_freq_drop_pct": round((1 - full_freq / base_freq) * 100, 1),
    }
    outpath = "/root/Fast-LeWorldModel/root_cause_diag.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {outpath}", flush=True)

    npu_pred.release()
    npu_enc.release()

if __name__ == "__main__":
    main()
