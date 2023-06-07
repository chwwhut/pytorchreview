# 逻辑斯蒂回归
import torch.nn
import numpy as np
import matplotlib.pyplot as plt

# 1.prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 将sigmoid函数应用到结果中
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 定义MSE(均方差)损失函数，size_average=False不求均值
criterion = torch.nn.BCELoss(reduction='mean')
# optim优化模块的SGD，第一个参数就是传递权重，model.parameters() model的所有权重
# 优化器对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # loss为一个对象，但会自动调用__str__()所以不会出错
    print(epoch, loss.item())

    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 根据梯度和预先设置的学习率进行更新
    optimizer.step()

# 打印权重和偏置值,weight是一个值但是一个矩阵
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# 测试
x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred=', y_test.data)

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view(200, 1)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
