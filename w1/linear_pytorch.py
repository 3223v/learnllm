# w1/linear_pytorch.py

import torch
import matplotlib.pyplot as plt

# ---------- 设备自动检测 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

