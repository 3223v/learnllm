import numpy as np

# 创建
a = np.array([1,2,3])
b = np.array(a)
c = np.array([[1,2,3],[2,33,89]])
d = np.array(19)
e = np.array([19])
print(a,b,c,d,e)
print(d is e)

# 基础操作

a = np.array([[1,2,3,4,5,6,7],[23,31,44,5,6,45,9]])

print(a[-1],a[1])

print(a[0,0])
#属性

print(a.shape)

print(a.size)

print(a.dtype)


# 其他创建

a = np.zeros(2)
b = np.ones(3)
c = np.empty(3)
print(a,a.dtype,b,c)