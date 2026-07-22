import torch

print(torch.__version__)

print(torch.backends.mps.is_available()) # M芯片应当输出 True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)