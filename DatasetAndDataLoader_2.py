import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDateset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        # [-1]加中括号拿出来是矩阵，不加是向量
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):  # 实例化对象后，该类能支持下标操作，通过index拿出数据
        return self.x_data[item], self.y_data[item]  # 返回的是元组（x,y)

    def __len__(self):
        return self.len


dataset = MyDateset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)
# windows中使用并行进程要封装代码 if __name__ == '__main__':


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 这是nn下的Sigmoid是一个模块没有参数，在function调用的Sigmoid是函数
        self.sigmoid = torch.nn.Sigmoid()
        # self.activate = torch.nn.ReLU()  error all elements of input should be between 0 and 1

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader):  # for i, (inputs,labels) in enumerate(train_loader,0)
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

"""
torchvision.datasets
MNIST Fashion-MNIST EMNIST COCO LSUN ImageFolader ....

四步：准备数据集-设计模型-构建损失函数和优化器-周期训练
"""

"""

"""