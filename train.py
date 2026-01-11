# -*- coding: utf-8 -*-
"""
train.py

目的：
- 读取 X/y（npy）
- 切分 train/val/test（按 config 比例）
- 构造DataLoader
- 初始化模型（ParallelCNNBiLSTMModel）
- 训练：loss = MSE（按你当前losses/rul_loss.py）
- 验证：记录val_loss（用于EarlyStopping 和保存best_model）
- 调度器：ReduceLROnPlateau（论文常用策略）
- 输出：best_model.pt + 训练日志 train_log.csv

运行：
cd D:\PY\vs_new
python train.py
"""

import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from config import Config, ensure_out_dir


# =========================================================
# 0) 解决“直接运行脚本时的包导入问题”
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT：工程根目录 vs_new

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # 把工程根目录加入 sys.path，保证from models... / from losses... 可用


# =========================================================
# 1) 导入模型与损失（你已经有 models/ 和 losses/）
# =========================================================
from models.fusion_model import ParallelCNNBiLSTMModel
from losses.run_loss import get_loss_fn, RMSELoss, nasa_score
from data.dataset import load_xy_from_npy, WindowTimeSeriesDataset


# =========================================================
# 2) 一些通用工具函数
# =========================================================
def set_seed(seed: int) -> None:
    """
    set_seed：固定随机种子，增强可复现性
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """
    pick_device：自动选择训练设备（优先GPU）
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    split_indices：把样本索引打乱并切分为 train/val/test
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test = 剩余

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_loader(
    dataset: WindowTimeSeriesDataset,
    idx: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int
):
    """
    make_loader：基于 WindowTimeSeriesDataset 子集构造DataLoader
    """
    subset = Subset(dataset, indices=idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return loader


class EarlyStopping:
    """
    EarlyStopping：验证集 loss 多轮不提升则停止训练
    """
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """
        返回 True 表示需要early stop
        """
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


def run_one_epoch(model, loader, optimizer, criterion, device, train: bool, use_amp: bool):
    """
    run_one_epoch：跑一个epoch（训练或验证）
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_hat = model(xb)
            loss = criterion(y_hat, yb)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def main():
    cfg = Config()
    ensure_out_dir(cfg)
    set_seed(cfg.seed)

    device = pick_device()
    print(f"[Device] {device}")

    # =========================
    # 1) 读取 X/y
    # =========================
    X, y = load_xy_from_npy(cfg.x_npy_path, cfg.y_npy_path)

    # 基本形状检查（避免你拿错文件）
    if X.ndim != 3:
        raise ValueError(f"X 维度应为 3（[N,L,F]），但拿到{X.shape}")
    if y.ndim != 2:
        raise ValueError(f"y 维度应为 2（[N,D]），但拿到{y.shape}")

    N, L, F = X.shape
    D = y.shape[1]

    print(f"[Data] X.shape={X.shape}, y.shape={y.shape}")

    # 维度对齐检查（必须与config 一致）
    if L != cfg.window_size:
        raise ValueError(f"window_size 不一致：X 的L={L}，config.window_size={cfg.window_size}")
    if F != cfg.in_features:
        raise ValueError(f"in_features 不一致：X 的F={F}，config.in_features={cfg.in_features}")
    if D != cfg.out_dim:
        raise ValueError(f"out_dim 不一致：y 的D={D}，config.out_dim={cfg.out_dim}")

    # =========================
    # 2) 切分 train/val/test
    # =========================
    train_idx, val_idx, test_idx = split_indices(
        n=N,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed
    )

    dataset = WindowTimeSeriesDataset(X, y)
    train_loader = make_loader(dataset, train_idx, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = make_loader(dataset, val_idx, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = make_loader(dataset, test_idx, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"[Split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # =========================
    # 3) 初始化模型
    # =========================
    model = ParallelCNNBiLSTMModel(in_features=cfg.in_features, out_dim=cfg.out_dim).to(device)

    # =========================
    # 4) 损失函数 / 优化器 / 调度器 / 早停
    # =========================
    criterion = get_loss_fn(cfg.loss_name)  # 默认 "mse"
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.min_lr,

    )

    early_stopper = EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_min_delta)

    # 评估指标（训练时可同时打印RMSE / NASA score，便于观察）
    rmse_fn = RMSELoss().to(device)

    # =========================
    # 5) 训练循环
    # =========================
    best_val = float("inf")

    # 写训练日志表头
    if not os.path.exists(cfg.train_log_path):
        with open(cfg.train_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_rmse,val_nasa,lr,sec\n")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            use_amp=cfg.use_amp
        )

        # 验证：用 MSE 做val_loss（与训练一致）
        val_loss = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            use_amp=cfg.use_amp
        )

        # 额外算RMSE / NASA score（便于你观察趋势；不是反传用）
        model.eval()
        with torch.no_grad():
            all_pred = []
            all_true = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_hat = model(xb)
                all_pred.append(y_hat)
                all_true.append(yb)

            y_pred = torch.cat(all_pred, dim=0)
            y_true = torch.cat(all_true, dim=0)

            val_rmse = float(rmse_fn(y_pred, y_true).item())
            val_nasa = float(nasa_score(y_pred, y_true).item())

        # 调度器依据val_loss 调整 lr
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        sec = time.time() - t0

        # 打印日志（按 log_every 控制）
        if epoch % cfg.log_every == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_rmse={val_rmse:.6f} | val_nasa={val_nasa:.6f} | lr={lr_now:.2e} | {sec:.1f}s"
            )

        # 写入 csv 日志
        with open(cfg.train_log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{val_rmse:.8f},{val_nasa:.8f},{lr_now:.8e},{sec:.3f}\n")

        # 保存 best_model（以 val_loss 最小为准）
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "config": cfg.__dict__,
                },
                cfg.ckpt_path
            )
            print(f"  ✅Save best model: epoch={epoch}, best_val={best_val:.6f} -> {cfg.ckpt_path}")

        # EarlyStopping 判断
        if early_stopper.step(val_loss):
            print(f"  ⚠EarlyStopping triggered at epoch={epoch}. best_val={early_stopper.best:.6f}")
            break

    # =========================
    # 6) 训练结束后，给一个test 快速评估（可选）
    # =========================
    print("\n[Train Done] Loading best checkpoint for quick test evaluation...")

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # test 上算 MSE/RMSE/NASA
    with torch.no_grad():
        all_pred = []
        all_true = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_hat = model(xb)
            all_pred.append(y_hat)
            all_true.append(yb)

        y_pred = torch.cat(all_pred, dim=0)
        y_true = torch.cat(all_true, dim=0)

        test_mse = float(get_loss_fn("mse")(y_pred, y_true).item())
        test_rmse = float(rmse_fn(y_pred, y_true).item())
        test_nasa = float(nasa_score(y_pred, y_true).item())

    print(f"[Test] mse={test_mse:.6f} | rmse={test_rmse:.6f} | nasa={test_nasa:.6f}")
    print("✅train.py finished")


if __name__ == "__main__":
    main()
