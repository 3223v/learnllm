import seaborn as sns
from pandas import plotting 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

data = load_iris()

# 将数据转换为DataFrame格式
df = pd.DataFrame(data.data, columns=data.feature_names)

# 添加品种标签
df['Species'] = data.target

# 查看数据集信息
print(f"数据集信息：\n{df.info()}")

# 查看前五条数据
print(f"前五条数据：\n{df.head()}") 

# 查看统计摘要
df.describe()

# 转换标签
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# 分离特征和标签
X = df.drop(columns="Species")
y = df["Species"]
feature_names = X.columns
labels = y.unique()

# 划分训练集和测试集
X_train, test_x, y_train, test_lab = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建并训练模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 输出决策树的文本表示
text_representation = tree.export_text(model)
print(text_representation)

# 绘制决策树
plt.figure(figsize=(30, 10), facecolor='g')
tree.plot_tree(model, feature_names=feature_names, class_names=labels, 
               rounded=True, filled=True, fontsize=14)