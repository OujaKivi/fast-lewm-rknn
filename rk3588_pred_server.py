#!/usr/bin/env python3
"""RK3588 Predictor NPU Server - 使用已有的 NPUPredictor 类"""
import sys
import json
import time
import base64
import numpy as np
import torch

sys.path.insert(0, '/root/Fast-LeWorldModel')
from npu_predictor import NPUPredictor

PREDICTOR_FP16_PATH = '/root/Fast-LeWorldModel/predictor_S300_pretrained_fp16.rknn'
PREDICTOR_INT8_PATH = '/root/Fast-LeWorldModel/predictor_S300_pretrained_int8.rknn'

def log(msg):
    print(msg, file=sys.stderr, flush=True)

class PredictorNPU:
    def __init__(self, mode='int8'):
        self.mode = mode
        model_path = PREDICTOR_INT8_PATH if mode == 'int8' else PREDICTOR_FP16_PATH
        log(f"[PredServer] Loading {mode} model: {model_path}")
        self.predictor = NPUPredictor(model_path)
        log(f"[PredServer] {mode} model loaded successfully")

    def inference(self, latent, act_emb):
        """
        多步预测：循环调用 5 次单步预测
        latent: (B, 1, 192) numpy
        act_emb: (B, 5, 192) numpy
        返回: (B, 5, 192) numpy
        """
        B = latent.shape[0]
        latent_t = torch.from_numpy(latent.astype(np.float32))
        
        preds = []
        current_latent = latent_t
        for t in range(5):
            act_emb_t = torch.from_numpy(act_emb[:, t:t+1, :].astype(np.float32))
            pred = self.predictor(current_latent, act_emb_t)  # (B, 1, 192)
            preds.append(pred)
            current_latent = pred
        
        pred_emb = torch.cat(preds, dim=1).numpy()  # (B, 5, 192)
        return pred_emb

    def release(self):
        self.predictor.release()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='int8', choices=['fp16', 'int8'])
    args = parser.parse_args()

    predictor = PredictorNPU(mode=args.mode)
    log(f"[PredServer] Ready. Mode={args.mode}. Waiting for requests...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            latent = np.frombuffer(base64.b64decode(request['latent']), dtype=np.float32).reshape(request['shape_latent'])
            act_emb = np.frombuffer(base64.b64decode(request['act_emb']), dtype=np.float32).reshape(request['shape_act_emb'])
            t0 = time.time()
            pred_emb = predictor.inference(latent, act_emb)
            infer_time = time.time() - t0
            response = {
                'pred_emb': base64.b64encode(pred_emb.astype(np.float32).tobytes()).decode('utf-8'),
                'shape_pred': list(pred_emb.shape),
                'infer_time': infer_time,
            }
            print(json.dumps(response), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}), flush=True)

if __name__ == '__main__':
    main()
