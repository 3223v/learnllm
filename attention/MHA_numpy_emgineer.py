# attention/MHA_numpy.py

# B：batch size

# S：序列长度

# d_model：模型维度

# H：注意力头数

# d_k = d_model / H：每个头的维度

# Q/K/V：query、key、value

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        # 分头数量和一个token对应的向量维度的关系，必须是整除的
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 初始化一个随机数生成器
        rng = np.random.default_rng(seed)
        # 创建正态分布中均值为0，方差0.02的矩阵，长宽都是d_model
        self.W_q = rng.normal(0, 0.02, (d_model, d_model))
        self.W_k = rng.normal(0, 0.02, (d_model, d_model))
        self.W_v = rng.normal(0, 0.02, (d_model, d_model))
        self.W_o = rng.normal(0, 0.02, (d_model, d_model))
        # 偏置b
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def _split_heads(self, x):
        B, S, D = x.shape
        x = x.reshape(B, S, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        B, H, S, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)      # (B, S, H, d_k)
        return x.reshape(B, S, H * d_k)  # (B, S, d_model)

    def forward(self, query, key, value, mask=None):

        Q = query @ self.W_q +self.b_q
        K = key @ self.W_k + self.b_k    
        V = value @ self.W_v + self.b_v

        # 计算完后分头

        Q = self._split_heads(Q)  # (B, H, Sq, d_k)
        K = self._split_heads(K)  # (B, H, Sk, d_k)
        V = self._split_heads(V)  # (B, H, Sk, d_k)
        # 计算 scores
        scores = Q @ K.transpose(0, 1, 3, 2)  # (B, H, Sq, Sk)
        scores = scores / np.sqrt(self.d_k)

        # mask
        if mask is not None:
            # mask 为 True 的位置保留，False 的位置变成极小值
            scores = np.where(mask, scores, -1e9)

        # softmax 得到注意力权重
        attn = softmax(scores, axis=-1)  # (B, H, Sq, Sk)

        # 加权 value
        out = attn @ V  # (B, H, Sq, d_k)

        # 合并多头
        out = self._combine_heads(out)  # (B, Sq, d_model)

        # 输出线性层
        out = out @ self.W_o + self.b_o  # (B, Sq, d_model)

        return out, attn