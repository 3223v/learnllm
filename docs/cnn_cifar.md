# 1. torchvision.datasets.CIFAR10 — 参数详解
CIFAR10：32×32 RGB彩色图，10分类；训练集50000张，测试集10000张，通道数=3
```python
torchvision.datasets.CIFAR10(
    root,
    train=True,
    transform=None,
    target_transform=None,
    download=False
)
```
|参数|含义|
|---|---|
|`root`|数据集存放的**根文件夹路径**；数据会放在root/cifar-10-batches-py|
|`train`|`True`=加载**训练集**；`False`=加载**测试集**|
|`transform`|**对图片做预处理**（PIL图像→Tensor、归一化、随机裁剪等），常用`transforms.Compose`组合操作；只改图像，不改标签|
|`target_transform`|**对标签做变换**，比如把数字标签转one-hot，分类任务很少用|
|`download`|`True`：root没数据就自动下载；已有数据不会重复下载|

✅ 最简示例：
```python
from torchvision import datasets, transforms

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
```
> 取出一条样本返回 `(image_tensor, label)`，image shape `(3,32,32)`

---

# 2. torch.nn.Conv2d — 重点4个参数 in_channels / out_channels / kernel_size / padding
```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1)
```
> 输入张量格式：`(N, C_in, H, W)` 【batch,通道,高,宽】
> 输出张量格式：`(N, C_out, H_out, W_out)`

### ① `in_channels`：输入通道数
输入特征图有多少个通道。
- RGB图片：`in_channels=3`（R/G/B三个通道）
- 灰度图：`in_channels=1`
- 上一层卷积输出16个特征图 → 当前层`in_channels=16`

> 理解：通道=特征图数量，每个通道代表一类特征。CIFAR10输入in_channels固定=3。

### ② `out_channels`：输出通道数 = 卷积核数量
**你准备用多少个卷积核，out_channels就等于几**。
- `out_channels=16`：16个独立卷积核，每个核提取一类特征（边缘、纹理等）
- 每个卷积核输出一张特征图，所以输出就有16个通道。
> 卷积核本身shape：`(out_channels, in_channels, k, k)`

### ③ `kernel_size`：卷积核大小
卷积窗口尺寸，可以写整数或元组。
- `kernel_size=3` → 3×3卷积核（CNN最常用）
- `kernel_size=(5,3)` → 高5，宽3的矩形卷积核

卷积核在特征图上滑动，做内积运算提取局部特征。

### ④ `padding`：边缘零填充
在图像/特征图**四周补0**。
- `padding=0`：不填充（valid卷积），输出长宽会缩小
- `padding=1`：上下左右各补一圈0
> 经典规则：`kernel_size=3, padding=1, stride=1` → **输出高宽不变（same卷积）**

📐 输出尺寸公式（记住这个）
$$
H_{out}=\lfloor \frac{H_{in}+2p-k}{s}+1\rfloor
$$
- $p$=padding，$k$=kernel_size，$s$=stride

示例：CIFAR输入 `H=32,W=32`
`nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1)`
$H_{out}= (32+2*1-3)/1+1 =32$ → 长宽不变，通道由3→16

---

# 3. torch.nn.MaxPool2d — 池化，怎么降维
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```
> 池化**不改变通道数量！只压缩高、宽（空间维度）**

### 原理
和卷积一样滑动窗口，**但没有可学习参数**。窗口内所有像素，**只保留最大值**作为输出。
- 最常用：`kernel_size=2, stride=2`，2×2窗口，步长2，窗口不重叠
- 效果：**高、宽全部减半**，通道数保持不变。

例子：输入 `(N,16,32,32)`，经过`MaxPool2d(2,2)`
输出：`(N,16,16,16)`
✅ 通道16不变；H,W从32→16，**空间尺寸降维**，减少计算量、保留强特征、增加平移不变性。

> 对比卷积：卷积可以改变通道数；池化**通道不动，只压缩空间分辨率**。

📌 举个直观小例子
输入4×4单通道：
```
1  2  3  4
5  6  7  8
9 10 11 12
13 14 15 16
```
2×2最大池化，stride=2：
- 左上2×2窗口`[1,2,5,6]` → max=6
- 右上`[3,4,7,8]` → max=8
- 左下`[9,10,13,14]` → max=14
- 右下`[11,12,15,16]` → max=16

输出2×2：
```
6   8
14 16
```

---

# 串联小CNN例子（CIFAR10），一次性串起来理解
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [batch, 3, 32,32] CIFAR输入
        x = self.conv1(x)    # [B,16,32,32] 通道变16，长宽不变
        x = self.pool(x)     # [B,16,16,16] 池化降维，H,W减半
        x = self.conv2(x)    # [B,32,16,16] 通道升到32
        x = self.pool(x)     # [B,32,8,8] 再次降维
        return x
```
