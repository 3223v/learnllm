# w1/linear_pytorch.py

import torch
import matplotlib.pyplot as plt
# ---------- 设备自动检测 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


torch.manual_seed(42)

X = torch.rand(100, 1, device=device) * 10

t_w = 2.0

t_b = 1.0

y = t_w * X + t_b + torch.rand(100, 1, device=device) * 1.5

w = torch.randn(1, 1, device=device, requires_grad=True)

b = torch.randn(1, device=device, requires_grad=True)

lr = 0.01

epochs = 500


def forward(X, w, b):
    return X @ w + b


def mse_loss(y_p, y_t):
    return ((y_p - y_t) ** 2).mean()


l_h = []


print(
    f"初始 w={w.item():.4f}, b={b.item():.4f}, Loss={mse_loss(forward(X,w,b), y).item():.4f}"
)

for epoch in range(epochs):

    y_p = forward(X, w, b)

    loss = mse_loss(y_p, y)

    l_h.append(loss.item())

    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

    if (epoch*1)%50 ==0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

print(f"最终 w={w.item():.4f}, b={b.item():.4f}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X.cpu().numpy(), y.cpu().numpy(), color='blue', label='真实值')
plt.scatter(X.cpu().numpy(), forward(X, w, b).detach().cpu().numpy(), color='red', alpha=0.6, label='预测值')
plt.legend()
plt.title('拟合效果')

plt.subplot(1, 2, 2)
plt.plot(l_h)
plt.xlabel('迭代次数')
plt.ylabel('MSE Loss')
plt.title('损失下降曲线')
plt.show()