import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 1.prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 对图像进行处理 convert the PIL Image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # mnist 灰度图，只有单通道
])

"""
Normalize
神经网络特别钟爱经过标准化处理后的数据。
标准化处理指的是，data减去它的均值，再除以它的标准差，最终data将呈现均值为0方差为1的数据分布。
神经网络模型偏爱标准化数据，原因是均值为0方差为1的数据在sigmoid、tanh经过激活函数后求导得到的导数很大，
反之原始数据不仅分布不均（噪声大）而且数值通常都很大（本例中数值范围是0~255），
激活函数后求导得到的导数则接近与0，这也被称为梯度消失。

还需要保持train_set、val_set和test_set标准化系数的一致性。
标准化系数就是计算要用到的均值和标准差，在本例中是((0.1307,), (0.3081,))，
均值是0.1307，标准差是0.3081，这些系数都是数据集提供方计算好的数据

因为mnist数据值都是灰度图，所以图像的通道数只有一个，因此均值和标准差各一个。
要是imagenet数据集的话，由于它的图像都是RGB图像，因此他们的均值和标准差各3个，分别对应其R,G,B值。
例如([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])就是Imagenet dataset的标准化系数
（RGB三个通道对应三组系数）。数据集给出的均值和标准差系数，每个数据集都不同的，都是数据集提供方给出的。

"""

train_dataset = datasets.MNIST(root='./dataset/mnist',  # ./当前文件夹下  ../上一级目录  /根目录
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset/mnist',
                              train=False,
                              download=False,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# 2.design model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # (N,1,28,28)->(N,784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 交叉熵损失函数CrossEntropyLoss，最后一层不用激活函数


model = Net()

# 3.construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
"""
CrossEntropyLoss<==>LogSoftmax+NLLLoss
"""
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# train and test 封装成函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # 不然tensor会构建计算图
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试不需要计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test()


