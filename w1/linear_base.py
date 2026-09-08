# w1/linear_base.py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(100,1)*10

y = 2.0 * X + 1.0 + np.random.rand(100,1)*1.5

w = np.random.randn(1,1)

b = np.zeros((1,))

def forward(X,w,b):

    return w*X+b

def mse_loss(y_true, y_pred):

    return np.sum((y_true-y_pred)**2)/y.shape[0]

y_pred = forward(X, w, b)
loss = mse_loss(y, y_pred)
print(f"初始损失: {loss.item():.4f}")  # 如果是numpy标量，用float()转换

# 可视化当前预测（红线）和真实数据（蓝点）
plt.scatter(X, y, color='blue')                                                                     
plt.scatter(X, y_pred, color='red', alpha=0.5) 
plt.show()


w = np.random.randn(1,1)

b = np.zeros((1,))

# 超参数

lr =0.01

epochs = 5000

loss_history = []

print(f"训练开始前，初始 w={w[0,0]:.4f}, b={b[0]:.4f}, Loss={mse_loss(y, forward(X,w,b)).item():.4f}")

for epoch in range(epochs):

    y_p = forward(X,w,b)

    loss = mse_loss(y,y_p)

    loss_history.append(loss.item())

    # 反向传播求梯度

    grad_p = (2/y.shape[0]) * (y_p - y)

    dw = np.dot(X.T, grad_p)
    
    db = np.sum(grad_p) 


    # 更新参数（梯度下降）
    w = w - lr * dw
    b = b - lr * db

    # 每 50 轮打印一次
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.4f}, w: {w[0,0]:.4f}, b: {b[0]:.4f}")

print(f"\n训练完成！最终 w={w[0,0]:.4f} (目标 2.0), b={b[0]:.4f} (目标 1.0)")


# 绘制最终拟合效果
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='真实值')
plt.scatter(X, forward(X, w, b), color='red', alpha=0.6, label='预测值')
plt.legend()
plt.title('最终拟合效果')

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('MSE Loss')
plt.title('损失下降曲线')
plt.show()