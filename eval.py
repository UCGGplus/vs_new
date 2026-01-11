# -*- coding: utf-8 -*-
"""
eval.py

目的：
- 加载训练得到的 best_model.pt
- 在 test 集上做评估（MSE / RMSE / NASA score）
- 保存预测结果到 outputs/eval_predictions.csv（便于你可视化/检查）

运行：
cd D:\\PY\\vs_new
python eval.py
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config, ensure_out_dir

# =========================================================
# 0) 解决“直接运行脚本时的包导入问题”
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.fusion_model import ParallelCNNBiLSTMModel
from losses.run_loss import get_loss_fn, RMSELoss, nasa_score


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_xy_npy(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"未找到 X npy：{x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"未找到 y npy：{y_path}")
    return np.load(x_path), np.load(y_path)


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_loader(X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, num_workers: int):
    X_sub = torch.tensor(X[idx], dtype=torch.float32)
    y_sub = torch.tensor(y[idx], dtype=torch.float32)
    ds = TensorDataset(X_sub, y_sub)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


def main():
    cfg = Config()
    ensure_out_dir(cfg)

    device = pick_device()
    print(f"[Device] {device}")

    # =========================
    # 1) 读取数据并切分 test
    # =========================
    X, y = load_xy_npy(cfg.x_npy_path, cfg.y_npy_path)
    N = X.shape[0]

    _, _, test_idx = split_indices(
        n=N,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed
    )

    test_loader = make_loader(X, y, test_idx, cfg.batch_size, cfg.num_workers)
    print(f"[Eval] test samples={len(test_idx)}")

    # =========================
    # 2) 初始化模型并加载 best checkpoint
    # =========================
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"未找到模型权重：{cfg.ckpt_path}（请先运行 train.py）")

    model = ParallelCNNBiLSTMModel(in_features=cfg.in_features, out_dim=cfg.out_dim).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[Eval] Loaded checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")

    # =========================
    # 3) 计算指标 + 保存预测结果
    # =========================
    mse_fn = get_loss_fn("mse")
    rmse_fn = RMSELoss().to(device)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_hat = model(xb)
            all_pred.append(y_hat)
            all_true.append(yb)

    y_pred = torch.cat(all_pred, dim=0)
    y_true = torch.cat(all_true, dim=0)

    test_mse = float(mse_fn(y_pred, y_true).item())
    test_rmse = float(rmse_fn(y_pred, y_true).item())
    test_nasa = float(nasa_score(y_pred, y_true).item())

    print(f"[Test Metrics] mse={test_mse:.6f} | rmse={test_rmse:.6f} | nasa={test_nasa:.6f}")

    # =========================
    # 4) 保存预测结果到 CSV（便于你人工检查）
    # =========================
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # 生成列名：pred_x1..pred_x8 / true_x1..true_x8
    pred_cols = [f"pred_x{i+1}" for i in range(y_pred_np.shape[1])]
    true_cols = [f"true_x{i+1}" for i in range(y_true_np.shape[1])]

    df_pred = pd.DataFrame(y_pred_np, columns=pred_cols)
    df_true = pd.DataFrame(y_true_np, columns=true_cols)

    df_out = pd.concat([df_true, df_pred], axis=1)
    df_out.to_csv(cfg.eval_pred_path, index=False, encoding="utf-8")

    print(f"✅ Predictions saved to: {cfg.eval_pred_path}")
    print("✅ eval.py finished")


if __name__ == "__main__":
    main()
