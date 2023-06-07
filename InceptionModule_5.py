import torch
from matplotlib import pyplot as plt
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
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)  # W,H不变

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)  # W,H不变
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]  # (W,H)相同，在channel 维度拼接 dim = 1 （B,C,W,H)
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))  # (W,H)   (28,28)->(12,12)  C 10
        x = self.incep1(x)  # 不变 C 88
        x = F.relu(self.mp(self.conv2(x)))  # (W,H)   (12,12)->(4,4)  C 20
        x = self.incep2(x)  # 不变 C 88
        x = x.view(batch_size, -1)  # （batch_size,88x4x4)
        x = self.fc(x)  # (1408->10)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# convert parameters and buffers of all modules to CUDA Tensor
model.to(device)

# 3.construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
"""
CrossEntropyLoss<==>LogSoftmax+NLLLoss
"""
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 4.train and test 封装成函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # 不然tensor会构建计算图
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test(accuracy):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试不需要计算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # send the inputs and targets at every step to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct / total)
    print('accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    accuracy = []
    for epoch in range(10):
        train(epoch)
        test(accuracy)

    epoch_list = range(10)
    plt.plot(epoch_list, accuracy)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

