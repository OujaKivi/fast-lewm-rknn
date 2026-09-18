#!/usr/bin/env python3
"""
RK3588 完整规划 Server
- ViT 编码 (CPU)
- Action Encoder (CPU)
- Predictor (CPU 或 NPU)
- CEM 采样 (CPU)
通信：stdin/stdout JSON + base64 图像
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

sys.path.insert(0, '/root/Fast-LeWorldModel')
from module import ActionPrefixEmbedder, ARPredictor, MLP
from npu_predictor import NPUPredictor

# 配置
WEIGHTS_PATH = '/root/Fast-LeWorldModel/weights/full_model_state.pt'
PREDICTOR_INT8_PATH = '/root/Fast-LeWorldModel/predictor_S300_pretrained_int8.rknn'
PREDICTOR_FP16_PATH = '/root/Fast-LeWorldModel/predictor_S300_pretrained_fp16.rknn'

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


class FastLeWMPlanner:
    def __init__(self, mode='cpu'):
        self.mode = mode
        self.device = 'cpu'
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
            transformer_depth=3, transformer_heads=3, transformer_dim_head=64,
        )
        ae_state = {k.replace('action_encoder.impl.', '', 1): v for k, v in state_dict.items() if k.startswith('action_encoder.impl.')}
        self.action_encoder.load_state_dict(ae_state)
        self.action_encoder.eval()
        self.action_encoder.requires_grad_(False)

        # 3. Predictor
        if mode == 'npu':
            self.predictor = NPUPredictor(PREDICTOR_INT8_PATH)
            self.predictor_type = 'npu'
        else:
            self.predictor = ARPredictor(
                depth=6, mlp_dim=2048, input_dim=192, hidden_dim=192,
                output_dim=192, heads=8, dim_head=128,
            )
            pred_state = {k.replace('predictor.', '', 1): v for k, v in state_dict.items() if k.startswith('predictor.')}
            self.predictor.load_state_dict(pred_state)
            self.predictor.eval()
            self.predictor.requires_grad_(False)
            self.predictor_type = 'cpu'

        # 4. Projector
        self.projector = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d)
        proj_state = {k.replace('projector.', '', 1): v for k, v in state_dict.items() if k.startswith('projector.')}
        self.projector.load_state_dict(proj_state)
        self.projector.eval()
        self.projector.requires_grad_(False)

        # 5. Pred Proj
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

        log(f"[Planner] Model loaded successfully in {mode} mode")

    def encode_image(self, img_b64):
        """解码 base64 图像并编码"""
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            output = self.encoder(img_tensor, interpolate_pos_encoding=True)
            cls_token = output.last_hidden_state[:, 0]  # (1, 192)
            emb = self.projector(cls_token)  # (1, 192)
        return emb

    def predict(self, emb, act_emb):
        """预测未来 latent"""
        if self.predictor_type == 'npu':
            # NPU 单步预测，循环 5 次
            B = emb.shape[0]
            current = emb.unsqueeze(1) if emb.dim() == 2 else emb  # (B, 1, 192)
            preds = []
            for t in range(5):
                act_t = act_emb[:, t:t+1, :] if act_emb.dim() == 3 else act_emb
                pred = self.predictor(current, act_t)  # (B, 1, 192)
                preds.append(pred)
                current = pred
            pred_emb = torch.cat(preds, dim=1)  # (B, 5, 192)
            return pred_emb
        else:
            # CPU 多步预测
            with torch.no_grad():
                preds = self.predictor(emb, act_emb)
                if preds.dim() == 2:
                    preds = self.pred_proj(preds)
                elif preds.dim() == 3:
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
        emb = current_emb.expand(N, -1)  # (N, 192)

        # Action encoder: 把 (N, 1, 50) 分成 5 个 block，每个 10 维
        # action_candidates: (N, 1, 50) -> (N, 5, 10)
        actions = action_candidates.reshape(N, 5, 10)

        with torch.no_grad():
            act_emb = self.action_encoder(actions, latent=emb.unsqueeze(1))  # (N, 5, 192)

        # Predict
        pred_emb = self.predict(emb, act_emb)  # (N, 5, 192)

        # Cost: 最后一步预测和 goal 的 MSE
        pred_last = pred_emb[:, -1:, :]  # (N, 1, 192)
        goal_expanded = goal_emb.unsqueeze(0).expand(N, 1, -1)  # (N, 1, 192)

        cost = F.mse_loss(pred_last, goal_expanded, reduction='none').sum(dim=-1).squeeze(-1)  # (N,)
        return cost

    def cem_plan(self, current_emb, goal_emb):
        """CEM 规划，返回最优动作序列"""
        # 初始化动作分布
        mean = torch.zeros(1, HORIZON, ACTION_DIM)
        var = torch.ones(1, HORIZON, ACTION_DIM)

        best_cost = float('inf')
        best_action = None

        for step in range(N_STEPS):
            # 采样候选
            candidates = torch.randn(NUM_SAMPLES, HORIZON, ACTION_DIM)
            candidates = candidates * var + mean
            candidates[0] = mean  # 强制第一个为当前 mean

            # 计算 cost
            costs = self.get_cost(current_emb, goal_emb, candidates)  # (N,)

            # 选择 top-k
            topk_vals, topk_inds = torch.topk(costs, k=TOPK, dim=0, largest=False)
            topk_candidates = candidates[topk_inds]  # (K, 1, 50)

            # 更新分布
            mean = topk_candidates.mean(dim=0, keepdim=True)
            var = topk_candidates.std(dim=0, keepdim=True)

            # 记录最优
            if topk_vals[0] < best_cost:
                best_cost = topk_vals[0].item()
                best_action = topk_candidates[0].clone()

        return best_action.squeeze(0).numpy(), best_cost  # (1, 50) -> (50,), cost

    def plan(self, current_img_b64, goal_img_b64):
        """完整规划流程"""
        t0 = time.time()

        # 编码图像
        t_enc_start = time.time()
        current_emb = self.encode_image(current_img_b64)
        goal_emb = self.encode_image(goal_img_b64)
        t_enc = time.time() - t_enc_start

        # CEM 规划
        t_cem_start = time.time()
        action, cost = self.cem_plan(current_emb, goal_emb)
        t_cem = time.time() - t_cem_start

        t_total = time.time() - t0

        return {
            'action': action.tolist(),  # 50维
            'cost': cost,
            'time_total': t_total,
            'time_encode': t_enc,
            'time_cem': t_cem,
            'mode': self.mode,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cpu', choices=['cpu', 'npu'])
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--num-samples', type=int, default=300)
    args = parser.parse_args()

    planner = FastLeWMPlanner(mode=args.mode)
    planner.N_STEPS = args.cem_steps
    planner.NUM_SAMPLES = args.num_samples
    log(f"[Planner] Ready. Mode={args.mode}. Waiting for requests...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            current_img = request['current_image']
            goal_img = request['goal_image']

            result = planner.plan(current_img, goal_img)
            print(json.dumps(result), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}), flush=True)


if __name__ == '__main__':
    main()
