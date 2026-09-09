#w1/linear_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset , DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

X = torch.randn(100,1) * 10

y = 2.0 * X + 1.0 + torch.rand(100,1) * 1.5

X ,y = X.to(device),y.to(device)

model = nn.Linear(1,1).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 500

for epoch in range(epochs):
    for batch_X,batch_y in dataloader:
        optimizer.zero_grad()

        p = model(batch_X)

        l = criterion(p,batch_y)

        l.backward()

        optimizer.step()
    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            total_loss = criterion(model(X), y)
            print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}") 
print(f"最终 w={model.weight.item():.4f}, b={model.bias.item():.4f}")
