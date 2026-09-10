# w2/resnet_transfer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform_train = transforms.Compose([
    transforms.Resize(224),                # 先放大到224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 必须用 ImageNet 归一化！
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

model = torchvision.models.resnet18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# ========== 替换原来的冻结循环 ==========
# 先默认全部冻结
for param in model.parameters():
    param.requires_grad = False

# 然后“解冻”最后两个关键模块：全连接层 fc 和 最后一个残差块 layer4
for name, param in model.named_parameters():
    if "fc" in name or "layer4" in name:
        param.requires_grad = True

# ========== 替换优化器 ==========
# 让优化器只管理那些被解冻的参数（即 requires_grad=True 的参数）
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.0001,   # 微调时学习率要调小（是原来的 1/10），防止破坏预训练权重
    momentum=0.9
)

# ========== 增加训练轮次 ==========
epochs = 10  # 建议微调跑 10 轮，充分让 layer4 适应新数据

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(
        f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Test Acc: {acc:.2f}%"
    )
