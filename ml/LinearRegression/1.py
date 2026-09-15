import numpy as np  
import matplotlib.pyplot as plt   
from sklearn.linear_model import LinearRegression

def true_fun(x):
    return 1.7*x + 0.3 

np.random.seed(42)
n_samples = 30

x_train = np.sort(np.random.rand(n_samples))

print(x_train)

y_train = (true_fun(x_train) +np.random.randn(n_samples) * 0.05).reshape(n_samples,1)

print(y_train)

model = LinearRegression()  # 定义模型
model.fit(x_train[:, np.newaxis], y_train)  # 训练模型
print("输出参数w:", model.coef_)  # 输出模型参数w
print("输出参数b:", model.intercept_)  # 输出参数b

X_test = np.linspace(0, 1, 100)
print(X_test)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(x_train, y_train)  # 画出训练集的点
plt.legend(loc="best")
plt.show()