# -*- coding: utf-8 -*-
"""
losses/rul_loss.py

本文件目的：
- 定义 RUL / 回归任务中常用的损失函数与评估指标
- 同时兼容：
  1) 你当前任务：多维时间序列预测（y.shape = [B, 8]）
  2) 论文原始任务：RUL 单值预测（y.shape = [B, 1]）

包含内容：
- MSELoss（训练常用）
- RMSELoss（直观误差）
- NASA Score（RUL 论文常用评价指标）
"""

import torch
import torch.nn as nn


# ============================================================
# 1. MSE（均方误差）——你当前训练阶段的主力
# ============================================================
class MSELoss(nn.Module):
    """
    MSELoss（Mean Squared Error）

    定义：
        MSE = mean((y_hat - y)^2)

    适用：
    - 多维回归（如你现在的 8维预测）
    - 单维回归（RUL）

    特点：
    - 连续可导
    - 对大误差惩罚更大
    """

    def __init__(self, reduction: str = "mean"):
        """
        参数：
        - reduction: 'mean' | 'sum' | 'none'
          与 torch.nn.MSELoss 一致
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y_hat: [B, D]
        y:     [B, D]
        """
        diff = y_hat - y
        loss = diff ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


# ============================================================
# 2. RMSE（均方根误差）——评估更直观
# ============================================================
class RMSELoss(nn.Module):
    """
    RMSELoss（Root Mean Squared Error）

    定义：
        RMSE = sqrt(mean((y_hat - y)^2))

    特点：
    - 单位与原始数据一致
    - 常用于 evaluation / report
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps  # 防止 sqrt(0) 数值不稳定

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_hat - y) ** 2)
        rmse = torch.sqrt(mse + self.eps)
        return rmse


# ============================================================
# 3. NASA Score（论文 RUL 常用指标）
# ============================================================
def nasa_score(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    NASA Score（常见于 C-MAPSS / RUL 论文）

    定义（逐样本）：
        e = y_hat - y
        if e < 0:
            score = exp(-e / 13) - 1
        else:
            score = exp(e / 10) - 1

    特点：
    - 对“预测过早失效”和“预测过晚失效”惩罚不对称
    - 更贴近工程维护风险

    注意：
    - 严格意义上 NASA Score 是为“单值 RUL”设计的
    - 若 y 是多维（如你现在的 8维预测），这里会：
      👉 对每一维独立计算 score，再取 mean
    """

    # 确保是 float
    y_hat = y_hat.float()
    y = y.float()

    e = y_hat - y

    score = torch.zeros_like(e)

    # e < 0（预测偏小：过早）
    mask_neg = e < 0
    score[mask_neg] = torch.exp(-e[mask_neg] / 13.0) - 1.0

    # e >= 0（预测偏大：过晚）
    mask_pos = e >= 0
    score[mask_pos] = torch.exp(e[mask_pos] / 10.0) - 1.0

    # 若是多维，取 batch + feature 的平均
    return score.mean()


# ============================================================
# 4. 工具函数：根据名字返回 loss（给 train.py 用）
# ============================================================
def get_loss_fn(name: str):
    """
    根据字符串返回对应 loss / metric

    支持：
    - "mse"
    - "rmse"
    - "nasa"

    用法：
        loss_fn = get_loss_fn("mse")
        loss = loss_fn(y_hat, y)
    """
    name = name.lower()

    if name == "mse":
        return MSELoss()
    elif name == "rmse":
        return RMSELoss()
    elif name == "nasa":
        return nasa_score
    else:
        raise ValueError(f"Unsupported loss name: {name}")


# ============================================================
# 5. 自检入口（python losses/rul_loss.py）
# ============================================================
if __name__ == "__main__":

    print("=== rul_loss 自检开始 ===")

    B = 4
    D = 8

    y = torch.randn(B, D)
    y_hat = y + 0.1 * torch.randn(B, D)

    # MSE
    mse_fn = MSELoss()
    mse_val = mse_fn(y_hat, y)
    print("MSE =", mse_val.item())

    # RMSE
    rmse_fn = RMSELoss()
    rmse_val = rmse_fn(y_hat, y)
    print("RMSE =", rmse_val.item())

    # NASA score
    score_val = nasa_score(y_hat, y)
    print("NASA score =", score_val.item())

    print("✅ rul_loss 自检通过")
