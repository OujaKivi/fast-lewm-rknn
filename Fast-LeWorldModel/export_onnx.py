"""
Fast-LeWM 模型导出 ONNX（用于后续 RKNN 转换）。

导出两个模型：
  1. action_encoder.onnx  —— ActionPrefixEmbedder（输入: actions[S,5,2] + latent[S,1,192]，输出: act_emb[S,1,192]）
  2. predictor.onnx       —— ARPredictor（输入: latent[S,1,192] + act_emb[S,1,192]，输出: pred[S,1,192]）

固定 batch S=300（CEM 默认候选数），opset=12，常量折叠。
随机权重（仅用于转换和性能测试；真实推理需加载 HF 权重）。

用法：
  python export_onnx.py --outdir ./onnx_out --batch 300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from module import ActionPrefixEmbedder, ARPredictor

# ---- 超参（与 profile/bench 一致）----
EMBED_DIM = 192
ACTION_DIM = 2
ACTION_BLOCKS = 5
AP_DEPTH, AP_HEADS, AP_DIMHEAD, AP_MLP = 3, 6, 32, 768
PRED_DEPTH, PRED_VHEADS, PRED_VDIMHEAD, PRED_MLP, PRED_FUSION = 6, 16, 64, 2048, 768


class ActionEncoderWrapper(torch.nn.Module):
    """包装 ActionPrefixEmbedder，固定 return_last_only=True，便于 ONNX 导出。"""
    def __init__(self, act_enc):
        super().__init__()
        self.act_enc = act_enc

    def forward(self, actions, latent):
        # actions: (S, 5, 2), latent: (S, 1, 192)
        return self.act_enc(actions, return_last_only=True, latent=latent)


def build_models(device="cpu"):
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


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def export_onnx(model, dummy_args, path, input_names, output_names, opset=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        model, dummy_args, path,
        input_names=input_names, output_names=output_names,
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,  # 固定 shape
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  导出: {path}  ({size_mb:.2f} MB)")
    return path


def verify_onnx(path, torch_model, dummy_args, atol=1e-3):
    """用 onnxruntime 验证 ONNX 输出与 PyTorch 一致。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [跳过验证] onnxruntime 未安装")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = torch_model(*dummy_args)
    if isinstance(torch_out, (list, tuple)):
        torch_out = torch_out[0]
    feeds = {inp.name: arg.cpu().numpy() for inp, arg in zip(sess.get_inputs(), dummy_args)}
    onnx_out = sess.run(None, feeds)[0]
    diff = abs(torch_out.cpu().numpy() - onnx_out).max()
    print(f"  验证: max_abs_diff={diff:.6e}  {'OK' if diff < atol else 'WARN(>atol)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./onnx_out")
    ap.add_argument("--batch", type=int, default=300, help="固定 batch size（CEM 候选数）")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    S = args.batch
    print(f"=== Fast-LeWM ONNX 导出 (batch={S}, opset={args.opset}) ===")

    act_enc, predictor = build_models(device)
    print(f"  action_encoder 参数: {n_params(act_enc):,}")
    print(f"  predictor 参数: {n_params(predictor):,}")

    # ---- dummy 输入 ----
    actions = torch.randn(S, ACTION_BLOCKS, ACTION_DIM, device=device)
    latent = torch.randn(S, 1, EMBED_DIM, device=device)
    act_emb = torch.randn(S, 1, EMBED_DIM, device=device)

    # ---- 1. action encoder ----
    print("\n[1/2] ActionPrefixEmbedder")
    enc_wrapper = ActionEncoderWrapper(act_enc)
    enc_path = os.path.join(args.outdir, f"action_encoder_S{S}.onnx")
    export_onnx(enc_wrapper, (actions, latent), enc_path,
                 input_names=["actions", "latent"], output_names=["act_emb"], opset=args.opset)
    if not args.no_verify:
        verify_onnx(enc_path, enc_wrapper, (actions, latent))

    # ---- 2. predictor ----
    print("\n[2/2] ARPredictor")
    pred_path = os.path.join(args.outdir, f"predictor_S{S}.onnx")
    export_onnx(predictor, (latent, act_emb), pred_path,
                 input_names=["latent", "act_emb"], output_names=["pred"], opset=args.opset)
    if not args.no_verify:
        verify_onnx(pred_path, predictor, (latent, act_emb))

    print(f"\n=== 完成。ONNX 文件在 {os.path.abspath(args.outdir)}/ ===")
    for f in sorted(os.listdir(args.outdir)):
        if f.endswith(".onnx"):
            fp = os.path.join(args.outdir, f)
            print(f"  {f}  ({os.path.getsize(fp)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
