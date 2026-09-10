# w2/cnn_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 将 PIL 图像转为 Tensor，并归一化到 [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到 [-1, 1]
    ]
)

# download
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# DataLoader
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.fc1 = nn.Linear(32*8*8,128)

        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                 
        return x
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向 + 损失
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

    # 每个 epoch 结束后计算测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")

print("训练完成！")


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), test_accuracies, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('测试准确率变化')
plt.legend()
plt.show()