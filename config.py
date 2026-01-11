# -*- coding: utf-8 -*-
"""
config.py

本文件目的：
- 统一管理训练/评估用到的超参数与路径
- train.py / eval.py 只读取这里的配置，便于复现实验与调参

与你当前数据严格对齐：
- X: [N, 30, 8]
- y: [N, 8]
"""

from dataclasses import dataclass  # dataclass：把“配置类”写得更简洁
import os  # os：拼接路径、创建目录


@dataclass
class Config:
    # =========================
    # 1) 工程路径配置
    # =========================

    project_root: str = os.path.dirname(os.path.abspath(__file__))
    # project_root：当前工程根目录（也就是 vs_new）

    data_dir: str = os.path.join(project_root, "data", "datacsv")
    # data_dir：你的数据目录（你把csv/npy都放在 data/datacsv）

    x_npy_path: str = os.path.join(data_dir, "X_ws30_h1.npy")
    # x_npy_path：滑窗后的输入 X 保存位置

    y_npy_path: str = os.path.join(data_dir, "y_ws30_h1.npy")
    # y_npy_path：滑窗后的标签 y 保存位置

    out_dir: str = os.path.join(project_root, "outputs")
    # out_dir：输出目录（保存模型、日志、预测结果等）

    ckpt_path: str = os.path.join(out_dir, "best_model.pt")
    # ckpt_path：训练过程中保存“验证集最优模型”的权重

    train_log_path: str = os.path.join(out_dir, "train_log.csv")
    # train_log_path：保存训练日志（epoch、train_loss、val_loss、lr 等）

    eval_pred_path: str = os.path.join(out_dir, "eval_predictions.csv")
    # eval_pred_path：eval.py 保存预测与真实值对比（便于你检查）

    # =========================
    # 2) 数据维度配置（与你的数据对齐）
    # =========================

    window_size: int = 30
    # window_size：滑窗长度 L，你的 X 第二维就是 30

    in_features: int = 8
    # in_features：输入特征维 F，你的 X 第三维就是 8

    out_dim: int = 8
    # out_dim：输出维度，你的 y 是 8 维

    # =========================
    # 3) 切分比例（train/val/test）
    # =========================

    seed: int = 42
    # seed：随机种子，确保可复现

    train_ratio: float = 0.7
    # train_ratio：训练集比例

    val_ratio: float = 0.15
    # val_ratio：验证集比例

    test_ratio: float = 0.15
    # test_ratio：测试集比例（eval.py 默认在 test 上评估）

    # =========================
    # 4) 训练超参数（贴近论文风格：Adam + EarlyStopping + ReduceLROnPlateau）
    # =========================

    batch_size: int = 32
    # batch_size：批大小

    max_epochs: int = 50
    # max_epochs：最大训练轮数，EarlyStopping 会提前终止

    lr: float = 1e-3
    # lr：学习率（论文常用 1e-3 / 1e-4；你先用 1e-3）

    weight_decay: float = 0.0
    # weight_decay：权重衰减（L2 正则），默认先不开

    # loss 选择：你说“就上面的 loss”，所以默认用 mse
    loss_name: str = "mse"
    # loss_name：训练损失函数名，来自 losses/rul_loss.py 的 get_loss_fn
    # 可选："mse"（训练用）/ 你也可以改成别的，但目前按你要求用 mse

    # =========================
    # 5) 训练策略（EarlyStopping + ReduceLROnPlateau）
    # =========================

    early_stop_patience: int = 8
    # early_stop_patience：验证损失多少轮不提升就停止

    early_stop_min_delta: float = 1e-6
    # early_stop_min_delta：认为“提升”的最小差值

    lr_reduce_factor: float = 0.1
    # lr_reduce_factor：ReduceLROnPlateau 的 factor

    lr_reduce_patience: int = 3
    # lr_reduce_patience：验证损失多少轮不提升就降低 lr

    min_lr: float = 1e-6
    # min_lr：学习率下限

    # =========================
    # 6) 设备与加速
    # =========================

    use_amp: bool = False
    # use_amp：是否启用混合精度（先关闭更稳）

    num_workers: int = 0
    # num_workers：DataLoader 多进程加载数据；Windows 上新手建议 0

    log_every: int = 1
    # log_every：每多少个 epoch 打印一次日志


def ensure_out_dir(cfg: Config) -> None:
    """
    ensure_out_dir：确保输出目录存在
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
