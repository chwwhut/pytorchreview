import numpy as np
from matplotlib import pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# initial guess of weight
w = 1.0


# 定义模型
def forward(x):
    return x * w


# 定义loss函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# define the gradient function
def gradient(x, y):
    return 2 * x * (x * w - y)


print("Predict (before training)", 4, forward(4))

mse_list = []
# 计算loss
for epoch in range(100):
    l_sum = 0
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
        l_sum += l

    mse = l_sum/3
    mse_list.append(mse)
    print("progress:", epoch, "w=", w, "loss=", l)

print("Predict (after training)", 4, forward(4))

epoch_list = range(100)
plt.plot(epoch_list, mse_list)
plt.xlabel("epoch")
plt.ylabel("mse")
plt.show()