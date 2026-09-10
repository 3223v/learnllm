import os
# 新增这两行，放在最顶部
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = load_dataset("stanfordnlp/imdb")
print(f"训练集大小：{len(dataset['train'])}，测试集大小：{len(dataset['test'])}")


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


counter = Counter()

for example in dataset["train"]:
    counter.update(tokenize(example["text"]))

vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(10000):
    vocab[word] = len(vocab)
print(f"词表大小；{len(vocab)}")


class IMDBDataset(Dataset):
    def __init__(self, hf_data, vocab, max_len=200):
        self.data = hf_data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        tokens = tokenize(example["text"])

        indices = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

        if len(indices) > self.max_len:
            indices = indices[: self.max_len]
        else:
            indices = indices + [self.vocab["<pad>"]] * (self.max_len - len(indices))
        return torch.tensor(indices), torch.tensor(example["label"])


train_dataset = IMDBDataset(dataset["train"], vocab)
test_dataset = IMDBDataset(dataset["test"], vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2,
                 hidden_dim=128, num_classes=2, max_len=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 填空1: Embedding 层（和 LSTM 版一样，注意 padding_idx=0）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码（可学习参数，简单直接）
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # 填空2: Transformer 编码器层
        # 提示：nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
        #                                  dim_feedforward=hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dim_feedforward=hidden_dim,batch_first=True)

        # 填空3: 堆叠 num_layers 层
        # 提示：nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)

        # 分类头（和 LSTM 版一样，输入维度是 embed_dim，不再是 hidden_dim*2）
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()

        # 1. 词嵌入 + 位置编码相加
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        embedded = embedded + self.pos_embedding(positions)  # 广播相加

        # 2. 构建 padding mask（告诉 attention 忽略 <pad>）
        # src_key_padding_mask: (batch, seq_len)，True 表示该位置被 mask
        pad_mask = (x == 0)  # 填空4：判断哪些位置是 padding

        # 3. 通过 Transformer
        # 注意：TransformerEncoder 默认输入 (seq_len, batch, dim)，除非 batch_first=True
        out = self.transformer(embedded, src_key_padding_mask=pad_mask)
        # out: (batch, seq_len, embed_dim)

        # 4. 池化（对非 padding 位置取平均）
        # 简单起见，直接对时间维度取平均
        pooled = out.mean(dim=1)  # (batch, embed_dim)

        # 5. 分类
        out = self.dropout(pooled)
        out = self.fc(out)
        return out

model = SentimentTransformer(len(vocab)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 测试
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%"
    )
