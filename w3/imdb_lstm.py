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


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        # 填空1: 定义 Embedding 层，注意 padding_idx=0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 填空2: 定义 LSTM 层，batch_first=True, bidirectional=True
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 填空3: 双向 LSTM 的输出维度是 hidden_dim * 2，定义全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        pooled = lstm_out.mean(dim=1)  # 对时间维度做平均池化
        out = self.dropout(pooled)
        out = self.fc(out)
        return out


model = SentimentLSTM(len(vocab)).to(device)
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
