import numpy as np

# === 练习1: 广播机制验证 ===
# 不要猜结果，先写代码验证
a = np.array([[1, 2, 3]])       # shape (1, 3)
b = np.array([[10], [20], [30]]) # shape (3, 1)
c = a + b
print(c.shape, c)  # 思考：为什么不是报错？结果是什么？

# === 练习2: 向量化 vs 循环 ===
# 体验为什么深度学习必须用矩阵运算
x = np.random.randn(10000)

# 慢：Python循环
import time
start = time.time()
result_loop = [xi**2 for xi in x]
print(f"Loop: {time.time()-start:.4f}s")

# 快：NumPy向量化
start = time.time()
result_vec = x ** 2
print(f"Vectorized: {time.time()-start:.4f}s")
# 通常快100倍以上！

# === 练习3: 矩阵乘法维度对齐 ===
W = np.random.randn(4, 3)  # 权重矩阵
x = np.random.randn(3, 1)  # 输入向量
y = W @ x                   # 输出 shape?
print(y,y.shape)              # 必须是 (4,1)，否则后面全错