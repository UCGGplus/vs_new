# -*- coding: utf-8 -*-
"""
models/cnn_branch.py

本文件实现论文中的 CNN 分支（空间/局部退化特征提取）：
- 3 层 1D-CNN：Conv1 -> (ECA) -> Conv2 -> Conv3
- kernel_size=7，激活 ReLU（与论文一致）
- ECA（Efficient Channel Attention）放在第一层卷积后（论文指出该位置效果最佳）
- 全局平均池化（GlobalAveragePooling1D 等价实现）
- 全连接层：64 -> 16（与 Table 1 一致）
- Dropout：0.2 / 0.1（与 Table 1 一致；在卷积输出与FC之间使用）

输入张量形状（与你 dataset 输出一致）：
- x: [B, L, F]  (B=batch, L=窗口长度, F=特征数)

输出：
- feat: [B, 16]  （给 fusion_model 拼接用）
"""

import torch  # torch：PyTorch 主库
import torch.nn as nn  # nn：神经网络层
import torch.nn.functional as F  # F：函数式接口（激活、softmax等）


class ECALayer(nn.Module):
    """
    ECA（Efficient Channel Attention）模块（论文 Fig.4）

    核心思想：
    - 先对时间维做全局平均池化：把 [B, C, L] -> [B, C]
    - 再用 1D 卷积在“通道维”做局部交互：等价于对通道权重做轻量建模
    - 经过 sigmoid 得到每个通道的权重
    - 用权重对原特征做通道缩放：y = x * w

    注意：
    - 这里的“1D 卷积”是对通道维做卷积，因此输入会 reshape 成 [B, 1, C]
    - kernel_size 使用 ECA 论文常见的自适应规则（不影响你贴近论文的整体结构）
    """

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()  # 调用父类初始化

        self.channels = channels  # 保存通道数 C（这里等于卷积输出通道数）

        # 计算自适应卷积核大小 k（ECA 常用经验公式）
        # k = | (log2(C) + b) / gamma | ，并且取最近的奇数，保证对称性
        t = int(abs((torch.log2(torch.tensor(float(channels))).item() + b) / gamma))
        k = t if t % 2 == 1 else t + 1  # 若是偶数则+1变奇数
        k = max(k, 3)  # 保底至少 3，避免过小导致注意力退化

        # 在通道维做 1D 卷积：输入 [B,1,C] -> 输出 [B,1,C]
        self.conv = nn.Conv1d(
            in_channels=1,  # 输入通道固定为 1（因为我们把通道维当成序列长度）
            out_channels=1,  # 输出通道也为 1
            kernel_size=k,  # 自适应核大小
            padding=(k - 1) // 2,  # padding 保持长度不变
            bias=False,  # 通常不需要 bias
        )

        self.sigmoid = nn.Sigmoid()  # sigmoid：把权重压到 0~1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L]
        """

        # 1) 全局平均池化（沿时间维 L 求均值）
        # x.mean(dim=2): [B,C,L] -> [B,C]
        y = x.mean(dim=2)

        # 2) 把 [B,C] reshape 成 [B,1,C] 以便做 1D conv
        y = y.unsqueeze(1)

        # 3) 在通道维做局部交互卷积：仍保持 [B,1,C]
        y = self.conv(y)

        # 4) sigmoid 得到通道权重： [B,1,C]，值域 0~1
        y = self.sigmoid(y)

        # 5) 把权重 reshape 回 [B,C,1] 以便与 x 广播相乘
        y = y.squeeze(1).unsqueeze(2)

        # 6) 通道加权：x * y（广播到时间维）
        out = x * y

        return out


class CNNBranch(nn.Module):
    """
    论文 CNN 分支实现（Fig.3 + Table 1）

    - Conv1 (256, k=7) -> ReLU -> ECA
    - Conv2 (96,  k=7) -> ReLU
    - Conv3 (32,  k=7) -> ReLU
    - GlobalAveragePooling1D（沿时间维平均）得到 [B, 32]
    - Dense 64 -> ReLU -> Dropout(0.1)
    - Dense 16 -> ReLU
    """

    def __init__(
        self,
        in_features: int,     # 输入特征维 F（你的数据里是 8）
        dropout_conv: float = 0.2,  # 对应 Table 1 的 0.2
        dropout_fc: float = 0.1,    # 对应 Table 1 的 0.1
    ):
        super().__init__()  # 父类初始化

        # Conv1：输入通道=in_features，输出通道=256（Table 1）
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=256,
            kernel_size=7,
            padding=3,  # kernel=7 时 padding=3 可保持长度不变（same）
            bias=True,
        )

        # ECA：放在第一层卷积后（论文说明该位置效果最好）
        self.eca = ECALayer(channels=256)

        # Conv2：256 -> 96（Table 1）
        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=96,
            kernel_size=7,
            padding=3,
            bias=True,
        )

        # Conv3：96 -> 32（Table 1）
        self.conv3 = nn.Conv1d(
            in_channels=96,
            out_channels=32,
            kernel_size=7,
            padding=3,
            bias=True,
        )

        # ReLU 激活（论文 CNN 激活函数为 ReLU）
        self.relu = nn.ReLU()

        # Dropout（卷积特征后）
        self.dropout_conv = nn.Dropout(p=dropout_conv)

        # 全连接层：32 -> 64（Table 1）
        self.fc1 = nn.Linear(32, 64)

        # 全连接层：64 -> 16（Table 1）
        self.fc2 = nn.Linear(64, 16)

        # Dropout（FC 中间）
        self.dropout_fc = nn.Dropout(p=dropout_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        return: [B, 16]
        """

        # 1) Conv1d 需要形状 [B, C, L]，而你的输入是 [B, L, F]
        # 所以要转置：把特征维 F 作为通道 C
        x = x.transpose(1, 2)  # [B,L,F] -> [B,F,L]

        # 2) 第一层卷积 + ReLU
        x = self.conv1(x)      # [B,F,L] -> [B,256,L]
        x = self.relu(x)       # ReLU 非线性

        # 3) ECA 通道注意力（论文 Fig.4）
        x = self.eca(x)        # [B,256,L] -> [B,256,L]

        # 4) Conv2 + ReLU
        x = self.conv2(x)      # [B,256,L] -> [B,96,L]
        x = self.relu(x)

        # 5) Conv3 + ReLU
        x = self.conv3(x)      # [B,96,L] -> [B,32,L]
        x = self.relu(x)

        # 6) 卷积输出后做 Dropout（对应论文 dropout=0.2 的用法之一）
        x = self.dropout_conv(x)

        # 7) GlobalAveragePooling1D：沿时间维 L 做平均
        # x.mean(dim=2)：[B,32,L] -> [B,32]
        x = x.mean(dim=2)

        # 8) Dense 32->64 + ReLU
        x = self.fc1(x)        # [B,32] -> [B,64]
        x = self.relu(x)

        # 9) FC Dropout（对应论文 dropout=0.1）
        x = self.dropout_fc(x)

        # 10) Dense 64->16 + ReLU
        x = self.fc2(x)        # [B,64] -> [B,16]
        x = self.relu(x)

        return x


if __name__ == "__main__":
    # ===========================
    # 自检：python models/cnn_branch.py
    # ===========================

    # 1) 构造一个假的 batch 输入（与你数据一致：L=30，F=8）
    B = 4
    L = 30
    F_dim = 8
    x = torch.randn(B, L, F_dim)

    # 2) 初始化 CNN 分支
    model = CNNBranch(in_features=F_dim)

    # 3) 前向计算
    feat = model(x)

    # 4) 打印形状检查
    print("Input x.shape =", tuple(x.shape))      # 期望 (4,30,8)
    print("CNN feat.shape =", tuple(feat.shape))  # 期望 (4,16)

    # 5) 形状强校验（不符合直接报错）
    if feat.shape != (B, 16):
        raise ValueError("CNNBranch 输出形状不正确，请检查实现。")

    print("✅ CNNBranch 自检通过")
