#!/usr/bin/env python3
"""
RK3588 完整规划 Server
- ViT 编码 (CPU 或 NPU)
- Action Encoder (CPU)
- Predictor (CPU 或 NPU，terminal-only planning fast path)
- CEM 采样 (CPU)
通信：stdin/stdout JSON + base64 图像

更新记录：
- 2026-09-20: 对齐官方配置（action heads=6x32, predictor heads=16x64）
- 2026-09-20: 规划路径只预测 terminal latent，不再计算未使用的 5 个 horizon 输出
"""
import sys
import json
import time
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ViTModel, ViTConfig
from torchvision.transforms import v2 as transforms
from PIL import Image
import io
from pathlib import Path

# The service is pinned to four Cortex-A76 cores. Avoid running eight default
# PyTorch workers on that four-core affinity mask, which adds large scheduler
# variance to the CPU action encoder.
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

LOCAL_MODEL_DIR = Path(__file__).resolve().parent / 'Fast-LeWorldModel'
sys.path.insert(0, str(LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else '/root/Fast-LeWorldModel'))
from module import ActionPrefixEmbedder, ARPredictor, MLP

# 配置
WEIGHTS_PATH = '/root/Fast-LeWorldModel/weights/full_model_state.pt'
# 模型已融合 pred_proj，输入和输出均为 [B, 1, 192]。
PREDICTOR_B300_FP16_PATH = '/root/Fast-LeWorldModel/predictor_terminal_with_proj_b300_fp16.rknn'
VIT_FP16_PATH = '/root/Fast-LeWorldModel/vit_encoder_projected_fp16.rknn'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CEM 配置
NUM_SAMPLES = 300
N_STEPS = 30
TOPK = 30
HORIZON = 1
ACTION_DIM = 50  # packed: 25 steps × 2 dim


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class TerminalNPUPredictor:
    """
    Fast-LeWM planning terminal-only NPU 推理封装。
    输入: latent (B, 1, 192) + terminal act_emb (B, 1, 192)
    输出: projected terminal latent (B, 1, 192)
    支持固定 batch 模型（如 batch=300），自动分批推理
    """
    def __init__(self, model_path, batch_size=1, core_mask=None):
        from rknnlite.api import RKNNLite
        self.batch_size = batch_size
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0_1_2
        
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        
        log(f"[NPU] Terminal predictor loaded: {model_path} (batch={batch_size})")
    
    def __call__(self, latent, act_emb):
        """
        latent: (B, 1, 192) torch.Tensor
        act_emb: (B, 1, 192) torch.Tensor
        返回: (B, 1, 192) torch.Tensor
        """
        B = latent.shape[0]
        latent_np = latent.detach().cpu().numpy().astype(np.float32)
        act_emb_np = act_emb.detach().cpu().numpy().astype(np.float32)
        
        # 如果输入 batch 等于模型固定 batch，直接一次推理
        if B == self.batch_size:
            out = self.rknn.inference(inputs=[latent_np, act_emb_np])
            return torch.from_numpy(out[0])
        
        # 否则分批推理
        outputs = []
        for start in range(0, B, self.batch_size):
            end = min(start + self.batch_size, B)
            lat_batch = latent_np[start:end]  # (batch, 1, 192)
            act_batch = act_emb_np[start:end]  # (batch, 1, 192)
            
            # 如果最后一批不足 batch_size，需要 padding 或逐个推理
            if lat_batch.shape[0] < self.batch_size:
                # 逐个推理（小批量，效率低但准确）
                for i in range(lat_batch.shape[0]):
                    lat_i = lat_batch[i:i+1]
                    act_i = act_batch[i:i+1]
                    # 注意：这里用的是固定 batch 模型，不能直接传 batch=1
                    # 需要 padding 到 batch_size
                    lat_pad = np.zeros((self.batch_size, 1, 192), dtype=np.float32)
                    act_pad = np.zeros((self.batch_size, 1, 192), dtype=np.float32)
                    lat_pad[0] = lat_i[0]
                    act_pad[0] = act_i[0]
                    out = self.rknn.inference(inputs=[lat_pad, act_pad])
                    outputs.append(out[0][0:1])
            else:
                out = self.rknn.inference(inputs=[lat_batch, act_batch])
                outputs.append(out[0])
        
        result = np.concatenate(outputs, axis=0)  # (B, 1, 192)
        return torch.from_numpy(result)
    
    def release(self):
        self.rknn.release()


class NPUImageEncoder:
    """Fused ViT and projector graph for normalized NCHW images."""

    def __init__(self, model_path, core_mask=None):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0_1_2
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        log(f"[NPU] Projected ViT loaded: {model_path}")

    def __call__(self, image):
        image_np = image.detach().cpu().numpy().astype(np.float32)
        output = self.rknn.inference(
            inputs=[image_np], data_format=["nchw"]
        )[0]
        return torch.from_numpy(output)

    def release(self):
        self.rknn.release()


class FastLeWMPlanner:
    def __init__(self, mode='cpu'):
        self.mode = mode
        self.device = 'cpu'
        self.num_samples = NUM_SAMPLES
        self.n_steps = N_STEPS
        self.topk = TOPK
        self.warm_start = False
        self.adaptive_cem = False
        self.min_cem_steps = 8
        self.stop_patience = 3
        self.cost_rel_tol = 2e-3
        self.mean_delta_tol = 8e-2
        self.std_tol = 1.5e-1
        self.warm_start_std_floor = 0.25
        self.warm_start_std_inflation = 1.25
        self.previous_plan = None
        self.previous_std = None
        self.seed = 1234
        self.solve_index = 0
        log(f"[Planner] Loading model in {mode} mode...")

        # 加载权重
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)

        # 1. ViT
        vit_config = ViTConfig(
            hidden_size=192,
            num_attention_heads=3,
            num_hidden_layers=12,
            intermediate_size=768,
            patch_size=14,
            image_size=224,
            num_channels=3,
            add_pooling_layer=False,
        )
        self.encoder = ViTModel(vit_config)
        encoder_state = {k.replace('encoder.', '', 1): v for k, v in state_dict.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(encoder_state, strict=False)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # 2. Action Encoder
        self.action_encoder = ActionPrefixEmbedder(
            input_dim=10, emb_dim=192,
            use_latent_condition=True, latent_dim=192,
            transformer_depth=3, transformer_heads=6, transformer_dim_head=32,
            transformer_mlp_dim=768,
        )
        ae_state = {k.replace('action_encoder.impl.', '', 1): v for k, v in state_dict.items() if k.startswith('action_encoder.impl.')}
        self.action_encoder.load_state_dict(ae_state)
        self.action_encoder.eval()
        self.action_encoder.requires_grad_(False)

        # 3. Predictor（官方 terminal-only planning 快路）
        if mode == 'npu':
            # 使用 batch=300 的 NPU 模型，用于 CEM 批量推理
            self.predictor = TerminalNPUPredictor(PREDICTOR_B300_FP16_PATH, batch_size=300)
            self.predictor_type = 'npu'
            log("[Planner] Using fused terminal NPU predictor (batch=300, heads=16x64)")
        else:
            self.predictor = ARPredictor(
                depth=6, mlp_dim=2048, input_dim=192, hidden_dim=192,
                output_dim=192, value_heads=16, value_dim_head=64,
                action_fusion_hidden_dim=768, action_fusion_zero_init=True,
                token_processing="batch",
            )
            pred_state = {k.replace('predictor.', '', 1): v for k, v in state_dict.items() if k.startswith('predictor.')}
            self.predictor.load_state_dict(pred_state)
            self.predictor.eval()
            self.predictor.requires_grad_(False)
            self.predictor_type = 'cpu'

        self.image_encoder = NPUImageEncoder(VIT_FP16_PATH) if mode == 'npu' else None

        # 4. Projector
        self.projector = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d)
        proj_state = {k.replace('projector.', '', 1): v for k, v in state_dict.items() if k.startswith('projector.')}
        self.projector.load_state_dict(proj_state)
        self.projector.eval()
        self.projector.requires_grad_(False)

        # 5. Pred Proj（NPU 模型已包含 pred_proj，CPU 模式需要单独做）
        self.pred_proj = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d)
        pp_state = {k.replace('pred_proj.', '', 1): v for k, v in state_dict.items() if k.startswith('pred_proj.')}
        self.pred_proj.load_state_dict(pp_state)
        self.pred_proj.eval()
        self.pred_proj.requires_grad_(False)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=(224, 224)),
        ])

        # Profiling 统计
        self.stats = {
            'encode': 0, 'action_enc': 0, 'predict': 0,
            'cost': 0, 'cem_update': 0, 'count': 0,
        }

        log(f"[Planner] Model loaded successfully in {mode} mode")

    def encode_image(self, img_b64):
        """解码 base64 图像并编码"""
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            if self.image_encoder is not None:
                emb = self.image_encoder(img_tensor)
            else:
                output = self.encoder(img_tensor, interpolate_pos_encoding=True)
                cls_token = output.last_hidden_state[:, 0]  # (1, 192)
                emb = self.projector(cls_token)  # (1, 192)
        return emb

    def predict(self, emb, act_emb):
        """
        预测 terminal latent。
        emb: (B, 192) 或 (B, 1, 192)
        act_emb: (B, 1, 192)
        返回: (B, 1, 192)
        """
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)  # (B, 1, 192)

        if self.predictor_type == 'npu':
            # NPU 模型已融合 pred_proj。
            pred = self.predictor(emb, act_emb)
            return pred
        else:
            with torch.no_grad():
                preds = self.predictor(emb, act_emb)
                b, t, _ = preds.shape
                preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
                preds = rearrange(preds, "(b t) d -> b t d", b=b, t=t)
            return preds

    def get_cost(self, current_emb, goal_emb, action_candidates):
        """
        计算候选动作的 cost
        current_emb: (1, 192)
        goal_emb: (1, 192)
        action_candidates: (N, 1, 50) - N candidates, horizon=1, action_dim=50
        返回: (N,) cost
        """
        N = action_candidates.shape[0]

        # 扩展 current_emb 到所有 candidate
        # Match the official rollout's flatten(...).clone(): do not feed the
        # action encoder a stride-0 view created by expand().
        emb = current_emb.expand(N, -1).contiguous()  # (N, 192)

        # Action encoder: 把 (N, 1, 50) 分成 5 个 block，每个 10 维
        actions = action_candidates.reshape(N, 5, 10)

        t0 = time.time()
        with torch.no_grad():
            act_emb = self.action_encoder(
                actions, return_last_only=True, latent=emb.unsqueeze(1)
            )  # (N, 1, 192)
            if act_emb.shape != (N, 1, 192):
                raise RuntimeError(f"Unexpected terminal action embedding shape: {tuple(act_emb.shape)}")
        self.stats['action_enc'] += time.time() - t0

        # 规划 cost 只使用 terminal latent。
        t0 = time.time()
        pred_emb = self.predict(emb, act_emb)  # (N, 1, 192)
        if pred_emb.shape != (N, 1, 192):
            raise RuntimeError(f"Unexpected terminal prediction shape: {tuple(pred_emb.shape)}")
        self.stats['predict'] += time.time() - t0

        # Cost: 最后一步预测和 goal 的 MSE
        t0 = time.time()
        pred_last = pred_emb
        goal_expanded = goal_emb.unsqueeze(0).expand(N, 1, -1)  # (N, 1, 192)
        cost = F.mse_loss(pred_last, goal_expanded, reduction='none').sum(dim=-1).squeeze(-1)  # (N,)
        self.stats['cost'] += time.time() - t0

        return cost

    @staticmethod
    def _shift_packed_plan(plan, executed_actions):
        """Shift a packed sequence of 2-D actions for receding-horizon reuse."""
        if plan is None:
            return None
        shift = max(0, int(executed_actions)) * 2
        flat = plan.reshape(-1)
        if shift == 0:
            return flat.clone()
        if shift >= flat.numel():
            return None
        shifted = torch.empty_like(flat)
        shifted[:-shift] = flat[shift:]
        # A constant-tail prior is less discontinuous than appending zeros.
        shifted[-shift:] = flat[-2:].repeat((shift + 1) // 2)[:shift]
        return shifted

    def reset_planner_state(self):
        self.previous_plan = None
        self.previous_std = None
        self.solve_index = 0

    def cem_plan(self, current_emb, goal_emb, executed_actions=1, reset=False):
        """CEM 规划，返回最优动作序列"""
        # 初始化动作分布
        mean = torch.zeros(1, HORIZON, ACTION_DIM)
        var = torch.ones(1, HORIZON, ACTION_DIM)

        warm_started = False
        if reset:
            self.reset_planner_state()
        generator = torch.Generator(device='cpu').manual_seed(
            self.seed + self.solve_index
        )
        if self.warm_start:
            shifted = self._shift_packed_plan(self.previous_plan, executed_actions)
            if shifted is not None:
                mean.copy_(shifted.reshape_as(mean))
                shifted_std = self._shift_packed_plan(
                    self.previous_std, executed_actions
                )
                if shifted_std is not None:
                    var.copy_(
                        (shifted_std.reshape_as(var) * self.warm_start_std_inflation)
                        .clamp(min=self.warm_start_std_floor, max=1.0)
                    )
                warm_started = True

        best_cost = float('inf')
        best_action = None
        previous_iter_cost = None
        stable_steps = 0
        stop_reason = 'max_steps'
        trace = []
        best_cost_history = []

        for step in range(self.n_steps):
            # 采样候选
            candidates = torch.randn(
                self.num_samples, HORIZON, ACTION_DIM, generator=generator
            )
            candidates = candidates * var + mean
            candidates[0] = mean  # 强制第一个为当前 mean

            # 计算 cost
            costs = self.get_cost(current_emb, goal_emb, candidates)  # (N,)

            # 选择 top-k
            t0 = time.time()
            topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=0, largest=False)
            topk_candidates = candidates[topk_inds]  # (K, 1, 50)

            # 更新分布
            previous_mean = mean
            mean = topk_candidates.mean(dim=0, keepdim=True)
            var = topk_candidates.std(dim=0, keepdim=True)
            self.stats['cem_update'] += time.time() - t0

            # 记录最优
            if topk_vals[0] < best_cost:
                best_cost = topk_vals[0].item()
                best_action = topk_candidates[0].clone()

            iter_cost = topk_vals[0].item()
            mean_delta = torch.sqrt(torch.mean((mean - previous_mean) ** 2)).item()
            rms_std = torch.sqrt(torch.mean(var ** 2)).item()
            rel_improvement = None
            if previous_iter_cost is not None:
                rel_improvement = ((previous_iter_cost - iter_cost) /
                                   max(abs(previous_iter_cost), 1e-12))
                cost_stable = rel_improvement <= self.cost_rel_tol
                mean_stable = mean_delta <= self.mean_delta_tol
                stable_steps = stable_steps + 1 if cost_stable and mean_stable else 0
            previous_iter_cost = iter_cost
            best_cost_history.append(best_cost)
            trace.append({
                'step': step + 1,
                'best_cost': iter_cost,
                'relative_improvement': rel_improvement,
                'mean_delta_rms': mean_delta,
                'rms_std': rms_std,
            })

            if self.adaptive_cem and step + 1 >= self.min_cem_steps:
                if rms_std <= self.std_tol:
                    stop_reason = 'variance_converged'
                    break
                if stable_steps >= self.stop_patience:
                    stop_reason = 'cost_and_mean_stable'
                    break
                if len(best_cost_history) > self.stop_patience:
                    old_best = best_cost_history[-self.stop_patience - 1]
                    window_gain = ((old_best - best_cost) /
                                   max(abs(old_best), 1e-12))
                    if window_gain <= self.cost_rel_tol:
                        stop_reason = 'best_cost_plateau'
                        break

        # Preserve the deployed planner's selection semantics: execute the
        # lowest-cost sample seen across all iterations.
        selected_action = best_action.squeeze(0).clone()
        self.previous_plan = selected_action.reshape(-1)
        self.previous_std = var.squeeze(0).reshape(-1).clone()
        self.solve_index += 1
        metadata = {
            'warm_started': warm_started,
            'iterations_used': len(trace),
            'stop_reason': stop_reason,
            'trace': trace,
        }
        return selected_action.numpy(), best_cost, metadata

    def plan(self, current_img_b64, goal_img_b64, executed_actions=1, reset=False):
        """完整规划流程"""
        t0 = time.time()

        # 重置统计
        self.stats = {k: 0 for k in self.stats}
        self.stats['count'] = 1

        # 编码图像
        t_enc_start = time.time()
        current_emb = self.encode_image(current_img_b64)
        goal_emb = self.encode_image(goal_img_b64)
        t_enc = time.time() - t_enc_start
        self.stats['encode'] = t_enc

        # CEM 规划
        t_cem_start = time.time()
        action, cost, cem_metadata = self.cem_plan(
            current_emb, goal_emb,
            executed_actions=executed_actions,
            reset=reset,
        )
        t_cem = time.time() - t_cem_start

        t_total = time.time() - t0

        # 各模块耗时占比
        total_cem_inner = (self.stats['action_enc'] + self.stats['predict'] +
                           self.stats['cost'] + self.stats['cem_update'])

        return {
            'action': action.tolist(),  # 50维
            'cost': cost,
            'time_total': t_total,
            'time_encode': t_enc,
            'time_cem': t_cem,
            'mode': self.mode,
            'cem': cem_metadata,
            'profiling': {
                'encode': self.stats['encode'],
                'action_encoder': self.stats['action_enc'],
                'predictor': self.stats['predict'],
                'cost_calc': self.stats['cost'],
                'cem_update': self.stats['cem_update'],
                'total_cem_inner': total_cem_inner,
                'percentages': {
                    'encode': self.stats['encode'] / t_total * 100 if t_total > 0 else 0,
                    'action_encoder': self.stats['action_enc'] / t_total * 100 if t_total > 0 else 0,
                    'predictor': self.stats['predict'] / t_total * 100 if t_total > 0 else 0,
                    'cost_calc': self.stats['cost'] / t_total * 100 if t_total > 0 else 0,
                    'cem_update': self.stats['cem_update'] / t_total * 100 if t_total > 0 else 0,
                }
            }
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cpu', choices=['cpu', 'npu'])
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--warm-start', action='store_true')
    parser.add_argument('--adaptive-cem', action='store_true')
    parser.add_argument('--min-cem-steps', type=int, default=8)
    parser.add_argument('--stop-patience', type=int, default=3)
    parser.add_argument('--cost-rel-tol', type=float, default=2e-3)
    parser.add_argument('--mean-delta-tol', type=float, default=8e-2)
    parser.add_argument('--std-tol', type=float, default=1.5e-1)
    parser.add_argument('--warm-start-std-floor', type=float, default=0.25)
    parser.add_argument('--warm-start-std-inflation', type=float, default=1.25)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    planner = FastLeWMPlanner(mode=args.mode)
    planner.n_steps = args.cem_steps
    planner.num_samples = args.num_samples
    planner.topk = args.topk
    planner.warm_start = args.warm_start
    planner.adaptive_cem = args.adaptive_cem
    planner.min_cem_steps = args.min_cem_steps
    planner.stop_patience = args.stop_patience
    planner.cost_rel_tol = args.cost_rel_tol
    planner.mean_delta_tol = args.mean_delta_tol
    planner.std_tol = args.std_tol
    planner.warm_start_std_floor = args.warm_start_std_floor
    planner.warm_start_std_inflation = args.warm_start_std_inflation
    planner.seed = args.seed
    if not 1 <= planner.topk <= planner.num_samples:
        parser.error("--topk must be in [1, num-samples]")
    log(f"[Planner] Ready. Mode={args.mode}. CEM steps={args.cem_steps}. "
        f"Samples={args.num_samples}. Warm-start={args.warm_start}. "
        f"Adaptive={args.adaptive_cem}. Waiting for requests...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            current_img = request['current_image']
            goal_img = request['goal_image']

            result = planner.plan(
                current_img,
                goal_img,
                executed_actions=request.get('executed_actions', 1),
                reset=request.get('reset', False),
            )
            print(json.dumps(result), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}), flush=True)


if __name__ == '__main__':
    main()
