import torch
# 一维张量
t1 = torch.tensor([1, 2, 3])
# 二维张量
t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 注意区分：
# torch.tensor() → 复制数据，新建张量
# torch.Tensor() → 构造器，接收shape（容易踩坑，不推荐优先用）
bad = torch.Tensor(2,3)  # 传入数字=形状，初始化随机垃圾值，慎用！

# 全0、全1
zeros = torch.zeros(2, 3)
ones = torch.ones(3, 2)

# 单位矩阵
eye = torch.eye(4)

# 未初始化（内存原始值，慎用）
empty = torch.empty(2,2)

# 区间序列
arange = torch.arange(0, 10, step=2)   # [0,2,4,6,8]
linspace = torch.linspace(0, 1, 5)      # 均分5个点

# 随机张量
rand = torch.rand(2,3)       # [0,1)均匀分布
randn = torch.randn(2,3)     # 标准正态 N(0,1)
randint = torch.randint(0, 10, (2,2))