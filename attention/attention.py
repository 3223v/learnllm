import torch
import math

import torch
import math

# 二维矩阵版 缩放点积自注意力（无batch）
def attention_2d(query, key, value, dropout=None):
    # 二维输入形状：[N, D]
    d_k = query.size(-1)  # 取特征维度 D

    # 核心注意力计算：二维矩阵转置 + 矩阵乘法
    # [N,D] @ [D,N] = [N,N] 得到元素关联分数矩阵
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 对每行做Softmax，得到注意力权重分布
    p_attn = scores.softmax(dim=-1)

    # 可选 dropout 防过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 注意力加权Value，输出融合后特征 [N,D]
    out = torch.matmul(p_attn, value)
    return out, p_attn

# 标准缩放点积注意力
# Q K V 三者维度一样
def attention(query, key, value, dropout = None):
    # 获取Q、K的特征维度 d_k，相当于一个token的向量维度，一句话会背翻译成 L*D 的矩阵
    # 其中L是token数量，D 则是一个token的向量化后向量维度
    d_k = query.size(-1)

    # 1. 计算注意力分数 + 缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 最后一维 Softmax，得到概率权重
    p_attn = scores.softmax(dim = -1)

    # 3. 可选 Dropout 防过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 4. 权重加权 V 得到输出，同时返回注意力矩阵
    return torch.matmul(p_attn, value), p_attn