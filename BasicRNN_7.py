import torch

"""
# how to use rnn
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# 如果设成batch_first=True 就要把batch_size提前
# (seqlen, batchsize,inputsize) -> (batchsize,seqlen,inputsize)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print('output size:', out.shape)  # (seqlen,batchsize,hiddensize)
print('output:', out)
print('hidden size:', hidden.shape)  # (numberlayers,batchsize,hiddensize)
print('hidden:', hidden)
"""
"""
# how to use RNNCell
batch_size = 1   # 批量数
seq_len = 3  # 有几个输入队列x1,x2,x3
input_size = 4  # 每个输入是几维向量
hidden_size = 2  # 每个隐藏层是几维向量

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('input size:', input.shape)

    hidden = cell(input, hidden)

    print('outputs size:', hidden.shape)
    print(hidden)
"""
"""
# prepare Data
input_size = 4  # 字典
hidden_size = 4  # output_size 分类
batch_size = 1  # 样本

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 5行4列 convert indices into one_hot vector

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # (seqlen,bacthsize,inputsize)  # 改成torch.tensor会报错 ？
labels = torch.LongTensor(y_data).view(-1, 1)  # (seqlen,1)

# torch.Tensor()是Python类，更明确的说，是默认张量类型torch.FloatTensor()的别名，
# torch.Tensor([1,2]) 会调用Tensor类的构造函数__init__，生成单精度浮点类型的张量。
# 
# torch.tensor()仅仅是Python的函数，函数原型是：
# torch.tensor(data, dtype=None, device=None, requires_grad=False)
# 其中data可以是：list, tuple, array, scalar等类型。
# torch.tensor()可以从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor。



# design model
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size  # initial the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)  # provide initial hidden


net = Model(input_size, hidden_size, batch_size)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# training cycle
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('predicted string:', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)  # ht = cell(xt,ht-1)
        loss += criterion(hidden, label)  # cross-entropy的两个输入（预测的向量，真实的y值）二者的维度不同   yt one-hot编码
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))
"""

"""
注意RNN和RNNcell的cross-entropy的区别，在于

1. RNNCell是一个one-hot对应一个数字label
2. 而RNN是多个one-hot向量组成的序列，对应一个由多个label数字组成的label序列

"""

# prepare Data
input_size = 4  # 字典
hidden_size = 4  # output_size 分类
batch_size = 1  # 样本

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 5行4列 convert indices into one_hot vector

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # (seqlen,bacthsize,inputsize)  # 改成torch.tensor会报错 ？
labels = torch.LongTensor(y_data)


# design model
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size  # initial the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)  # (seq, batch, hidden) -> (sxb, hidden)


net = Model(input_size, hidden_size, batch_size)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# training cycle
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()  # 数组
    print('predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss=%.3f' % (epoch + 1, loss.item()))
