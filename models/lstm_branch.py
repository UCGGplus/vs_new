# -*- coding: utf-8 -*-
"""
models/lstm_branch.py

本文件实现论文中的 BiLSTM 分支（时间序列特征提取）：
- 两层 BiLSTM（hidden=96/96，Table 1）
- Multi-Head Attention（subspaces=5，Table 1）
- 残差连接：attention输出与 BiLSTM 序列输出 O 相加（论文描述）
- 然后做时间维池化（等价于把序列变成固定向量）
- 全连接层：64 -> 16（Table 1）
- Dropout：0.2/0.1（与 Table 1 对齐）

输入：
- x: [B, L, F]

输出：
- feat: [B, 16]
"""

import math  # math：用于 sqrt 等数学函数
import torch  # torch：PyTorch 主库
import torch.nn as nn  # nn：神经网络层
import torch.nn.functional as F  # F：函数式接口


class MultiHeadSelfAttentionResidual(nn.Module):
    """
    自定义 Multi-Head Self-Attention（支持 heads=5，并满足 residual）

    为什么不直接用 nn.MultiheadAttention？
    - PyTorch 的 nn.MultiheadAttention 要求 embed_dim 能被 num_heads 整除
    - 论文 Table 1 给的 subspaces=5
    - 但 BiLSTM 输出维度 d_model = 2*hidden = 192（hidden=96，双向）
    - 192 / 5 不是整数
    - 因此这里用“自定义 MHA”：
      1) 每个 head 用 head_dim=32（可用且常见）
      2) concat 后维度 = 5*32 = 160
      3) 再用 out_proj 线性映射回 d_model=192
      4) 这样就能 residual：out + O

    输入/输出：
    - 输入 O: [B, L, d_model]
    - 输出:  [B, L, d_model]（可与 O 相加）
    """

    def __init__(
        self,
        d_model: int,        # 输入特征维度（这里应为 192）
        num_heads: int = 5,  # 论文 subspaces=5
        head_dim: int = 32,  # 每个 head 的子空间维度（工程上常用）
        dropout: float = 0.1 # 对应论文 dropout=0.1 的用法之一
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        # total_dim = num_heads * head_dim：拼接后的维度
        self.total_dim = num_heads * head_dim

        # Q/K/V 投影：把 d_model -> total_dim
        self.q_proj = nn.Linear(d_model, self.total_dim, bias=True)
        self.k_proj = nn.Linear(d_model, self.total_dim, bias=True)
        self.v_proj = nn.Linear(d_model, self.total_dim, bias=True)

        # out_proj：把 total_dim -> d_model，用于 residual 对齐
        self.out_proj = nn.Linear(self.total_dim, d_model, bias=True)

        # dropout：对注意力权重或输出做随机失活
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        """
        O: [B, L, d_model]
        return: [B, L, d_model]
        """

        B, L, _ = O.shape  # 取 batch、序列长度

        # 1) 计算 Q/K/V： [B,L,d_model] -> [B,L,total_dim]
        Q = self.q_proj(O)
        K = self.k_proj(O)
        V = self.v_proj(O)

        # 2) reshape 成多头形式：
        # [B,L,total_dim] -> [B,L,num_heads,head_dim] -> [B,num_heads,L,head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) scaled dot-product attention：
        # attention_scores: [B,num_heads,L,L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4) softmax 得到注意力权重（沿最后一维L归一化）
        attn = torch.softmax(scores, dim=-1)

        # 5) dropout（让注意力更鲁棒）
        attn = self.dropout(attn)

        # 6) 加权求和得到每个 head 的输出：
        # out_head: [B,num_heads,L,head_dim]
        out_head = torch.matmul(attn, V)

        # 7) 拼接 heads：
        # [B,num_heads,L,head_dim] -> [B,L,num_heads,head_dim] -> [B,L,total_dim]
        out = out_head.transpose(1, 2).contiguous().view(B, L, self.total_dim)

        # 8) 映射回 d_model： [B,L,total_dim] -> [B,L,d_model]
        out = self.out_proj(out)

        # 9) dropout（对输出再做一次失活）
        out = self.dropout(out)

        return out


class BiLSTMBranch(nn.Module):
    """
    论文 BiLSTM 分支实现（2-layer BiLSTM + Multi-Head Attention + residual）

    - 两层 BiLSTM：hidden=96/96（Table 1），bidirectional=True
    - attention subspaces=5（Table 1），并 residual 到 BiLSTM 输出序列 O
    - 池化得到定长向量
    - Dense：64 -> 16（Table 1）
    """

    def __init__(
        self,
        in_features: int,         # 输入特征维度 F（你的数据里是 8）
        hidden_size: int = 96,    # Table 1：BiLSTM hidden units 96/96
        num_layers: int = 2,      # Table 1 体现为 96/96 -> 两层
        dropout_lstm: float = 0.2,# Table 1 dropout=0.2 的用法之一
        dropout_fc: float = 0.1,  # Table 1 dropout=0.1 的用法之一
        num_heads: int = 5,       # Table 1：subspaces=5
        head_dim: int = 32,       # 自定义MHA每头维度
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM：batch_first=True 表示输入输出都是 [B,L,*]
        # 注意：PyTorch 的 LSTM dropout 只在 num_layers>1 时生效
        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        # d_model = 2*hidden_size（双向拼接输出）
        self.d_model = 2 * hidden_size  # 这里是 192

        # Multi-Head Self-Attention（自定义，支持 heads=5）
        self.mha = MultiHeadSelfAttentionResidual(
            d_model=self.d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout_fc,  # attention dropout 用 0.1
        )

        # LayerNorm：残差后做归一化（工程上常用，提升稳定性）
        self.norm = nn.LayerNorm(self.d_model)

        # Dropout：残差后的序列特征再丢弃一部分
        self.dropout_after_attn = nn.Dropout(p=dropout_fc)

        # FC：d_model -> 64（Table 1：BiLSTM后全连接 64/16）
        self.fc1 = nn.Linear(self.d_model, 64)

        # FC：64 -> 16
        self.fc2 = nn.Linear(64, 16)

        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(p=dropout_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        return: [B, 16]
        """

        # 1) BiLSTM 输出序列 O： [B,L,F] -> [B,L,2*hidden]
        O, _ = self.bilstm(x)

        # 2) Multi-Head Attention：对 O 做自注意力，输出与 O 同维
        A = self.mha(O)

        # 3) Residual：论文描述“attention加权后与 O 残差连接”
        H = O + A

        # 4) LayerNorm：稳定训练（不改变形状）
        H = self.norm(H)

        # 5) Dropout：进一步防过拟合
        H = self.dropout_after_attn(H)

        # 6) 池化：把序列 [B,L,d_model] -> 向量 [B,d_model]
        # 论文常见做法是 GlobalAveragePooling（沿时间维平均）
        h = H.mean(dim=1)

        # 7) Dense d_model->64 + ReLU
        h = self.fc1(h)
        h = self.relu(h)

        # 8) Dropout
        h = self.dropout_fc(h)

        # 9) Dense 64->16 + ReLU
        h = self.fc2(h)
        h = self.relu(h)

        return h


if __name__ == "__main__":
    # ===========================
    # 自检：python models/lstm_branch.py
    # ===========================

    B = 4
    L = 30
    F_dim = 8
    x = torch.randn(B, L, F_dim)

    model = BiLSTMBranch(in_features=F_dim)

    feat = model(x)

    print("Input x.shape =", tuple(x.shape))          # 期望 (4,30,8)
    print("BiLSTM feat.shape =", tuple(feat.shape))   # 期望 (4,16)

    if feat.shape != (B, 16):
        raise ValueError("BiLSTMBranch 输出形状不正确，请检查实现。")

    print("✅ BiLSTMBranch 自检通过")
