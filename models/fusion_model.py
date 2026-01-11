# -*- coding: utf-8 -*-
"""
models/fusion_model.py

实现论文提出的并行融合模型：
- CNNBranch（CNN+ECA）提取空间特征 -> 16维
- BiLSTMBranch（BiLSTM+MHA）提取时间特征 -> 16维
- concat 拼接 -> 32维
- FC 网络压缩（论文描述：FC network used to minimize input size）
- 输出预测

注意：
- 论文输出是 RUL（1维）
- 你现在的 y 是预测“下一时刻 8维特征”，所以 out_dim=8
"""

import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .lstm_branch import BiLSTMBranch



class ParallelCNNBiLSTMModel(nn.Module):
    """
    论文并行 CNN-BiLSTM 双注意力融合模型（结构对齐论文）

    输入：
    - x: [B, L, F]

    输出：
    - y_hat: [B, out_dim]
    """

    def __init__(
        self,
        in_features: int,     # 输入特征维度 F（你现在是 8）
        out_dim: int,         # 输出维度（你现在是 8；如果未来改成RUL就设1）
        dropout_head: float = 0.1,  # 头部 dropout（对齐论文的 0.1）
    ):
        super().__init__()

        # 1) CNN 分支（输出 16维）
        self.cnn = CNNBranch(in_features=in_features)

        # 2) BiLSTM 分支（输出 16维）
        self.lstm = BiLSTMBranch(in_features=in_features)

        # 3) 融合后的特征维度：16 + 16 = 32
        fusion_dim = 16 + 16

        # 4) 论文描述“FC network used to minimize input size”
        # 这里给一个符合论文风格的 head：
        # 32 -> 64 -> out_dim
        self.fc1 = nn.Linear(fusion_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_head)
        self.fc_out = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        return: [B, out_dim]
        """

        # 1) CNN 分支提特征
        f_cnn = self.cnn(x)  # [B,16]

        # 2) BiLSTM 分支提特征
        f_lstm = self.lstm(x)  # [B,16]

        # 3) concat 拼接（论文明确写了 concatenate）
        f = torch.cat([f_cnn, f_lstm], dim=1)  # [B,32]

        # 4) FC 压缩
        h = self.fc1(f)  # [B,64]
        h = self.relu(h)
        h = self.dropout(h)

        # 5) 输出层
        y_hat = self.fc_out(h)  # [B,out_dim]

        return y_hat


if __name__ == "__main__":
    # ===========================
    # 自检：python models/fusion_model.py
    # ===========================

    B = 4
    L = 30
    F_dim = 8
    out_dim = 8

    x = torch.randn(B, L, F_dim)

    model = ParallelCNNBiLSTMModel(in_features=F_dim, out_dim=out_dim)

    y_hat = model(x)

    print("Input x.shape =", tuple(x.shape))          # 期望 (4,30,8)
    print("Output y_hat.shape =", tuple(y_hat.shape)) # 期望 (4,8)

    if y_hat.shape != (B, out_dim):
        raise ValueError("ParallelCNNBiLSTMModel 输出形状不正确，请检查实现。")

    print("✅ Fusion Model 自检通过")
