"""实验2+3：Predictor CPU vs NPU 简单对比"""
import sys
sys.path.insert(0, '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel')

import json
import time
import base64
import subprocess
import numpy as np
import torch
import stable_worldmodel as swm

CKPT_DIR = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel/weights'
RUN_NAME = 'Fast-lewm_pusht'
SSH_CONFIG = '/Users/wangjiwei/.ssh/config_rknn'
SSH_HOST = 'rk3588'
SERVER_SCRIPT = '/root/Fast-LeWorldModel/rk3588_pred_server.py'


def load_model():
    model = swm.policy.AutoCostModel(RUN_NAME, cache_dir=CKPT_DIR)
    model = model.to('cpu').eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    for m in model.modules():
        if isinstance(m, torch.nn.GRU):
            m.train()
    return model


class NPUPredictorClient:
    def __init__(self, mode='int8'):
        self.mode = mode
        cmd = ['ssh', '-F', SSH_CONFIG, SSH_HOST,
               f'/root/miniconda3/envs/fast-lewm/bin/python {SERVER_SCRIPT} --mode {self.mode}']
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
        time.sleep(10)
        print(f"  NPU {mode} server started")

    def inference(self, latent, act_emb):
        request = {
            'latent': base64.b64encode(latent.astype(np.float32).tobytes()).decode(),
            'act_emb': base64.b64encode(act_emb.astype(np.float32).tobytes()).decode(),
            'shape_latent': list(latent.shape),
            'shape_act_emb': list(act_emb.shape),
        }
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        for _ in range(20):
            line = self.process.stdout.readline().strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
                if 'error' in resp:
                    raise RuntimeError(resp['error'])
                pred = np.frombuffer(base64.b64decode(resp['pred_emb']), dtype=np.float32).reshape(resp['shape_pred'])
                return pred, resp.get('infer_time', 0)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("No valid response")

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


def main():
    print("=" * 60)
    print("实验2+3：Predictor CPU vs NPU 对比")
    print("=" * 60)

    print("\n1. 加载 CPU 模型...")
    model = load_model()
    print("  模型加载成功")

    num_samples = 300
    num_runs = 10

    # 生成测试输入
    latent = torch.randn(num_samples, 1, 192)
    act_emb = torch.randn(num_samples, 5, 192)

    # CPU 推理
    print(f"\n2. CPU predictor 推理 (batch={num_samples})...")
    cpu_times = []
    with torch.no_grad():
        for i in range(num_runs):
            t0 = time.time()
            pred_cpu = model.predictor(latent, act_emb)
            cpu_times.append(time.time() - t0)
    cpu_mean = np.mean(cpu_times[2:]) * 1000
    print(f"  CPU: mean={cpu_mean:.1f}ms, std={np.std(cpu_times[2:])*1000:.1f}ms")

    # NPU INT8
    print(f"\n3. NPU INT8 predictor 推理 (batch={num_samples})...")
    npu_int8 = NPUPredictorClient('int8')
    int8_times = []
    int8_infer_times = []
    latent_np = latent.numpy()
    act_emb_np = act_emb.numpy()
    for i in range(num_runs):
        t0 = time.time()
        pred_int8, infer_time = npu_int8.inference(latent_np, act_emb_np)
        int8_times.append(time.time() - t0)
        int8_infer_times.append(infer_time)
    int8_mean = np.mean(int8_times[2:]) * 1000
    int8_infer_mean = np.mean(int8_infer_times[2:]) * 1000
    print(f"  NPU INT8: total={int8_mean:.1f}ms, infer={int8_infer_mean:.1f}ms, comm={int8_mean-int8_infer_mean:.1f}ms")

    # NPU FP16
    print(f"\n4. NPU FP16 predictor 推理 (batch={num_samples})...")
    npu_int8.close()
    time.sleep(2)
    npu_fp16 = NPUPredictorClient('fp16')
    fp16_times = []
    fp16_infer_times = []
    for i in range(num_runs):
        t0 = time.time()
        pred_fp16, infer_time = npu_fp16.inference(latent_np, act_emb_np)
        fp16_times.append(time.time() - t0)
        fp16_infer_times.append(infer_time)
    fp16_mean = np.mean(fp16_times[2:]) * 1000
    fp16_infer_mean = np.mean(fp16_infer_times[2:]) * 1000
    print(f"  NPU FP16: total={fp16_mean:.1f}ms, infer={fp16_infer_mean:.1f}ms, comm={fp16_mean-fp16_infer_mean:.1f}ms")

    # 精度对比
    print("\n5. 精度对比 (vs CPU FP32)...")
    pred_cpu_np = pred_cpu.numpy()

    for name, pred_np in [("INT8", pred_int8), ("FP16", pred_fp16)]:
        mae = np.mean(np.abs(pred_cpu_np - pred_np))
        rmse = np.sqrt(np.mean((pred_cpu_np - pred_np) ** 2))
        signal_power = np.mean(pred_cpu_np ** 2)
        noise_power = np.mean((pred_cpu_np - pred_np) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        cos_sim = np.mean(np.sum(pred_cpu_np * pred_np, axis=-1) /
                           (np.linalg.norm(pred_cpu_np, axis=-1) * np.linalg.norm(pred_np, axis=-1) + 1e-10))
        print(f"  {name}: MAE={mae:.6f}, RMSE={rmse:.6f}, SNR={snr:.1f}dB, CosSim={cos_sim:.4f}")

    # 汇总
    results = {
        'batch_size': num_samples,
        'cpu_ms': cpu_mean,
        'npu_int8_total_ms': int8_mean,
        'npu_int8_infer_ms': int8_infer_mean,
        'npu_int8_comm_ms': int8_mean - int8_infer_mean,
        'npu_fp16_total_ms': fp16_mean,
        'npu_fp16_infer_ms': fp16_infer_mean,
        'npu_fp16_comm_ms': fp16_mean - fp16_infer_mean,
        'speedup_int8_vs_cpu': cpu_mean / int8_mean,
        'speedup_fp16_vs_cpu': cpu_mean / fp16_mean,
        'int8_snr_db': float(10 * np.log10(np.mean(pred_cpu_np**2) / (np.mean((pred_cpu_np - pred_int8)**2) + 1e-10))),
        'fp16_snr_db': float(10 * np.log10(np.mean(pred_cpu_np**2) / (np.mean((pred_cpu_np - pred_fp16)**2) + 1e-10))),
    }

    output_path = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/pred_bench_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    print(f"  CPU FP32:          {results['cpu_ms']:.1f} ms")
    print(f"  NPU INT8 (total):  {results['npu_int8_total_ms']:.1f} ms (infer {results['npu_int8_infer_ms']:.1f} + comm {results['npu_int8_comm_ms']:.1f})")
    print(f"  NPU FP16 (total):  {results['npu_fp16_total_ms']:.1f} ms (infer {results['npu_fp16_infer_ms']:.1f} + comm {results['npu_fp16_comm_ms']:.1f})")
    print(f"  Speedup INT8:      {results['speedup_int8_vs_cpu']:.2f}x")
    print(f"  Speedup FP16:      {results['speedup_fp16_vs_cpu']:.2f}x")
    print(f"  INT8 SNR:          {results['int8_snr_db']:.1f} dB")
    print(f"  FP16 SNR:          {results['fp16_snr_db']:.1f} dB")
    print(f"\n结果已保存到 {output_path}")

    npu_fp16.close()


if __name__ == '__main__':
    main()
