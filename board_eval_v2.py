"""
板端完整评估脚本 v2：使用手动构建的模型 + 官方 CEMSolver
"""
import sys
sys.path.insert(0, '/root/Fast-LeWorldModel')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import json
import time
import numpy as np
import torch
import torch.nn as nn
import base64
import io
from PIL import Image
from transformers import ViTModel, ViTConfig
from module import ActionPrefixEmbedder, ARPredictor, MLP
from stable_worldmodel.solver import CEMSolver
from gymnasium.spaces import Box
from stable_worldmodel.envs.pusht.env import PushT

# 配置
WEIGHTS_PATH = '/root/Fast-LeWorldModel/weights/full_model_state.pt'
MAX_STEPS = 3
npu_rknn = None  # 全局 NPU 模型句柄（predictor）
vit_npu_rknn = None  # 全局 ViT NPU 模型句柄
ACTION_BLOCK = 25

# 动作反标准化参数
ACTION_MEAN = np.array([-0.0078, 0.0069])
ACTION_STD = np.array([0.2085, 0.2067])

def build_model():
    """手动构建模型（对齐test_official_cem.py）"""
    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
    
    vit_config = ViTConfig(hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
        intermediate_size=768, patch_size=14, image_size=224, add_pooling_layer=False)
    encoder = ViTModel(vit_config)
    enc_state = {k.replace('encoder.', '', 1): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_state, strict=False)
    
    action_encoder = ActionPrefixEmbedder(
        input_dim=10, emb_dim=192, use_latent_condition=True, latent_dim=192,
        transformer_depth=3, transformer_heads=6, transformer_dim_head=32,
        transformer_mlp_dim=768,
    )
    ae_state = {k.replace('action_encoder.impl.', '', 1): v for k, v in state_dict.items() if k.startswith('action_encoder.impl.')}
    action_encoder.load_state_dict(ae_state)
    
    predictor = ARPredictor(
        depth=6, mlp_dim=2048, input_dim=192, hidden_dim=192, output_dim=192,
        value_heads=16, value_dim_head=64, action_fusion_hidden_dim=768,
        action_fusion_zero_init=True, token_processing="batch",
    )
    pred_state = {k.replace('predictor.', '', 1): v for k, v in state_dict.items() if k.startswith('predictor.')}
    predictor.load_state_dict(pred_state)
    
    projector = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d)
    proj_state = {k.replace('projector.', '', 1): v for k, v in state_dict.items() if k.startswith('projector.')}
    projector.load_state_dict(proj_state)
    
    pred_proj = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d)
    pp_state = {k.replace('pred_proj.', '', 1): v for k, v in state_dict.items() if k.startswith('pred_proj.')}
    pred_proj.load_state_dict(pp_state)
    
    class CostModelWrapper(nn.Module):
        def __init__(self, encoder, action_encoder, predictor, projector, pred_proj):
            super().__init__()
            self.encoder = encoder
            self.action_encoder = action_encoder
            self.predictor = predictor
            self.projector = projector
            self.pred_proj = pred_proj
            self.eval()
            for m in self.modules():
                if isinstance(m, torch.nn.GRU):
                    m.train()
        
        def get_cost(self, info_dict, action_candidates):
            with torch.no_grad():
                B, S, H, D = action_candidates.shape
                pixels = info_dict['pixels'][:, 0]
                goal = info_dict['goal'][:, 0]
                
                # ViT 编码
                if vit_npu_rknn is not None:
                    # 异构模式：ViT 用 NPU
                    cur_img_np = pixels[:, 0].float().numpy().astype(np.float32)
                    cur_out_np = vit_npu_rknn.inference(inputs=[cur_img_np])
                    cur_emb = torch.from_numpy(cur_out_np[0])
                    
                    goal_img_np = goal[:, 0].float().numpy().astype(np.float32)
                    goal_out_np = vit_npu_rknn.inference(inputs=[goal_img_np])
                    goal_emb = torch.from_numpy(goal_out_np[0])
                else:
                    # CPU 模式
                    cur_out = self.encoder(pixels[:, 0].float(), interpolate_pos_encoding=True)
                    cur_emb = self.projector(cur_out.last_hidden_state[:, 0])
                    goal_out = self.encoder(goal[:, 0].float(), interpolate_pos_encoding=True)
                    goal_emb = self.projector(goal_out.last_hidden_state[:, 0])
                
                cur_emb = (
                    cur_emb.unsqueeze(1).expand(B, S, -1).reshape(B*S, -1).contiguous()
                )
                goal_emb = goal_emb.unsqueeze(1).expand(B, S, -1).reshape(B*S, -1)
                
                actions = action_candidates.reshape(B*S, H, D).reshape(B*S, 5, 10)
                
                # 官方 planning 快路：prefix encoder 看完整 5 blocks，
                # predictor 只计算 CEM cost 所需的 terminal token。
                act_emb = self.action_encoder(
                    actions, return_last_only=True, latent=cur_emb.unsqueeze(1)
                )
                if act_emb.shape != (B * S, 1, 192):
                    raise RuntimeError(f"Unexpected terminal action embedding shape: {tuple(act_emb.shape)}")

                if npu_rknn is not None:
                    latent_in = cur_emb.unsqueeze(1).numpy().astype(np.float32)
                    act_emb_in = act_emb.numpy().astype(np.float32)
                    outputs = npu_rknn.inference(inputs=[latent_in, act_emb_in])
                    # Terminal RKNN 模型已融合 pred_proj。
                    terminal_pred = torch.from_numpy(outputs[0])
                else:
                    pred = self.predictor(cur_emb.unsqueeze(1), act_emb)
                    terminal_pred = self.pred_proj(pred[:, 0]).unsqueeze(1)

                if terminal_pred.shape != (B * S, 1, 192):
                    raise RuntimeError(f"Unexpected terminal prediction shape: {tuple(terminal_pred.shape)}")
                
                cost = ((terminal_pred - goal_emb.unsqueeze(1))**2).sum(dim=-1).squeeze(-1)
                return cost.reshape(B, S)
    
    return CostModelWrapper(encoder, action_encoder, predictor, projector, pred_proj)

def load_npu_model():
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn('/root/Fast-LeWorldModel/predictor_terminal_with_proj_b300_fp16.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    return rknn

def load_npu_int8_model():
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn('/root/Fast-LeWorldModel/predictor_terminal_with_proj_b300_int8.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    return rknn

def load_vit_npu_model():
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn('/root/Fast-LeWorldModel/vit_encoder.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    return rknn

def decode_image(img_b64):
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(img)

def img_to_tensor(img):
    """图像转为 (1,1,3,224,224) tensor，ImageNet归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = (img_t - mean.squeeze(0)) / std.squeeze(0)
    # resize to 224x224
    import torch.nn.functional as F
    img_t = F.interpolate(img_t.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    return img_t.unsqueeze(0)  # (1,1,3,224,224)

def main():
    global npu_rknn, vit_npu_rknn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-cases', type=str, required=True)
    parser.add_argument('--mode', type=str, default='cpu')
    args = parser.parse_args()
    
    print(f"[BoardEval] Building model...", flush=True)
    model = build_model()
    print(f"[BoardEval] Model built", flush=True)
    
    npu_rknn = None
    vit_npu_rknn = None
    
    if args.mode == 'npu':
        print(f"[BoardEval] Loading NPU predictor model (FP16)...", flush=True)
        npu_rknn = load_npu_model()
        print(f"[BoardEval] NPU predictor model loaded", flush=True)
    elif args.mode == 'het':
        print(f"[BoardEval] Loading ViT NPU model (heterogeneous mode: ViT NPU + Predictor CPU)...", flush=True)
        vit_npu_rknn = load_vit_npu_model()
        print(f"[BoardEval] ViT NPU model loaded", flush=True)
    elif args.mode == 'het_int8':
        print(f"[BoardEval] Loading ViT NPU + Predictor INT8 NPU (full heterogeneous mode)...", flush=True)
        vit_npu_rknn = load_vit_npu_model()
        npu_rknn = load_npu_int8_model()
        print(f"[BoardEval] Full heterogeneous mode loaded (ViT NPU + Predictor INT8 NPU)", flush=True)
    
    # 创建CEMSolver
    solver = CEMSolver(model, batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device='cpu')
    
    class SimpleConfig:
        horizon = 1
        action_block = ACTION_BLOCK
        receding_horizon = 1
    
    action_space = Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32)
    solver.configure(action_space=action_space, n_envs=1, config=SimpleConfig())
    
    # 加载测试用例
    with open(args.test_cases, 'r') as f:
        test_cases = json.load(f)
    print(f"[BoardEval] Loaded {len(test_cases)} test cases", flush=True)
    
    results = []
    for ep_idx, tc in enumerate(test_cases):
        print(f"\n[BoardEval] Episode {ep_idx+1}/{len(test_cases)}", flush=True)
        
        env = PushT()
        env.reset()
        
        init_state = np.array(tc['init_state'])
        goal_state = np.array(tc['goal_state'])
        goal_img = decode_image(tc['goal_image'])
        
        env._set_state(init_state)
        env._set_goal_state(goal_state)
        
        goal_tensor = img_to_tensor(goal_img)
        
        success = False
        total_reward = 0
        plan_times = []
        
        for step in range(MAX_STEPS):
            cur_img = env.render()
            cur_tensor = img_to_tensor(cur_img)
            
            info_dict = {
                'pixels': cur_tensor,
                'goal': goal_tensor,
            }
            
            t0 = time.time()
            try:
                result = solver(info_dict)
                actions = result.get("actions", None)
                if actions is not None:
                    act_seq = actions.reshape(-1, 2)
                    if hasattr(act_seq, 'numpy'):
                        act_seq = act_seq.numpy()
                    else:
                        act_seq = np.array(act_seq)
                else:
                    act_seq = np.zeros((ACTION_BLOCK, 2))
            except Exception as e:
                print(f"  Step {step}: plan error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                act_seq = np.zeros((ACTION_BLOCK, 2))
            plan_time = time.time() - t0
            plan_times.append(plan_time)
            
            for a_step in range(ACTION_BLOCK):
                action = act_seq[a_step]
                action_unnorm = action * ACTION_STD + ACTION_MEAN
                action_unnorm = np.clip(action_unnorm, -1, 1)
                obs, reward, terminated, truncated, info = env.step(action_unnorm)
                total_reward += reward
                
                success, state_dist = env.eval_state(goal_state, env._get_obs())
                if success:
                    break
            
            if success:
                print(f"  Step {step}: SUCCESS!", flush=True)
                break
            
            if step % 5 == 0:
                print(f"  Step {step}: plan_time={plan_time:.2f}s, reward={total_reward:.1f}", flush=True)
        
        avg_plan_time = np.mean(plan_times) if plan_times else 0
        results.append({
            'episode': ep_idx,
            'success': success,
            'total_reward': float(total_reward),
            'steps': step + 1,
            'avg_plan_time': float(avg_plan_time),
        })
        print(f"  Result: success={success}, reward={total_reward:.1f}, steps={step+1}, avg_plan_time={avg_plan_time:.2f}s", flush=True)
    
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_plan_time = np.mean([r['avg_plan_time'] for r in results])
    
    summary = {
        'mode': args.mode,
        'num_episodes': len(results),
        'success_rate': success_rate,
        'avg_reward': float(avg_reward),
        'avg_plan_time': float(avg_plan_time),
        'results': results,
    }
    
    print(f"\n[BoardEval] Summary: success_rate={success_rate:.2%}, avg_reward={avg_reward:.1f}, avg_plan_time={avg_plan_time:.2f}s", flush=True)
    
    with open('/tmp/board_eval_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("[BoardEval] Results saved", flush=True)

if __name__ == '__main__':
    main()
