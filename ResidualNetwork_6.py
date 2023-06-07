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


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  # 保证输入张量维度与输出张量维度一样  x，y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
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
    for epoch in range(5):  # 过拟合严重 epoch增大
        train(epoch)
        test(accuracy)

    epoch_list = range(5)
    plt.plot(epoch_list, accuracy)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

