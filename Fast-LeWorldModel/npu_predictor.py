"""
NPUPredictor —— RKNN 版 ARPredictor 包装类。

接口与 module.ARPredictor 完全一致：
    predictor(latent, act_emb) -> pred
    latent:  (S, 1, 192) torch.Tensor
    act_emb: (S, 1, 192) torch.Tensor
    pred:    (S, 1, 192) torch.Tensor

内部用 rknn-toolkit-lite2 做 NPU 推理，自动处理 torch↔numpy 转换。
用于方案 A 异构调度：action_encoder 留 CPU，predictor 放 NPU。

用法：
    from npu_predictor import NPUPredictor
    npu_pred = NPUPredictor("./predictor_S300_fp16_v4.rknn")
    pred = npu_pred(latent_tensor, act_emb_tensor)
    npu_pred.release()
"""
import numpy as np
import torch
from rknnlite.api import RKNNLite


class NPUPredictor:
    """RKNN 版 predictor，接口兼容 torch.nn.Module 的 forward 调用。"""

    def __init__(self, rknn_path, core_mask=None, verbose=False):
        """
        Args:
            rknn_path: RKNN 模型文件路径
            core_mask: NPU 核选择，默认 NPU_CORE_AUTO
            verbose: 是否打印 RKNN 日志
        """
        self.rknn = RKNNLite(verbose=verbose)
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 失败: {rknn_path}, ret={ret}")

        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"初始化 NPU runtime 失败, ret={ret}")

        self.rknn_path = rknn_path
        self._call_count = 0

    def __call__(self, latent, act_emb):
        """
        推理一次 predictor。

        Args:
            latent:  torch.Tensor (S, 1, 192)，当前 latent
            act_emb: torch.Tensor (S, 1, 192)，action embedding

        Returns:
            pred: torch.Tensor (S, 1, 192)，预测的下一步 latent
        """
        # torch → numpy（RKNN 输入要求 float32，C 连续）
        latent_np = latent.detach().cpu().numpy().astype(np.float32, copy=False)
        act_emb_np = act_emb.detach().cpu().numpy().astype(np.float32, copy=False)

        # NPU 推理（输入正确时 inference 直接返回输出列表）
        outputs = self.rknn.inference(inputs=[latent_np, act_emb_np])

        if outputs is None or len(outputs) == 0:
            raise RuntimeError("NPU 推理返回空输出，请检查输入 shape 是否与模型一致")

        pred_np = outputs[0]  # (S, 1, 192) float32

        # numpy → torch，保持与输入相同的 device
        pred = torch.from_numpy(pred_np).to(latent.device)
        self._call_count += 1
        return pred

    def release(self):
        """释放 RKNN 资源。"""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    @property
    def call_count(self):
        return self._call_count
