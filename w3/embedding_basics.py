import torch
import torch.nn as nn

vocab = {
    "我": 0, 
    "爱": 1, 
    "学习": 2, 
    "深度": 3, 
    "学习2": 4,
    "<pad>": 5
}
vocab_size = len(vocab)

embed_dim = 8

embedding = nn.Embedding(vocab_size, embed_dim)

sentence = torch.tensor([0,1,3,2])

vectors = embedding(sentence)

print(f"词向量形状: {vectors.shape}")  # (4, 8)
print(f"第一个词的向量: {vectors[0]}")

# 5. 模拟一个 batch（3 个句子，长度不一，需要 padding）
# 句子1: [0,1,2]  句子2: [0,1,2,3]  句子3: [0,1]
batch = torch.tensor([
    [0, 1, 2, 5],  # 5 是 <pad>
    [0, 1, 2, 3],
    [0, 1, 5, 5],
])
batch_vectors = embedding(batch)
print(f"批次词向量形状: {batch_vectors.shape}")  # (3, 4, 8)