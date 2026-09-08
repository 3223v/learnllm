import numpy as np

a = np.array([[1, 2, 3]])       # shape (1, 3)
b = np.array([[10], [20], [30]]) # shape (3, 1)
c = a + b
print(c.shape, c)  # 广播机制，缺失的直接补齐

s = np.random.randn(4,2)
print(s)