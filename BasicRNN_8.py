import torch

# using embedding and linear layer
# prepare Data
num_class = 4
input_size = 4  # 字典大小？
hidden_size = 8
embedding_size = 10
batch_size = 1
num_layers = 2
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']  # 字典
x_data = [[1, 0, 2, 2, 3]]  # hello (batch, seq_len)
y_data = [3, 1, 2, 3, 2]  # ohlol (batch*seq_len)

inputs = torch.LongTensor(x_data)  # input should be long tensor
labels = torch.LongTensor(y_data)
"""
one-hot vectors：                         embedding vectors
1.high-dimension                          1.lower-dimension
2.sparse                                  2.dense
3.Hardcoded                               3.learned from data

在RNN模型的训练过程中，需要用到词嵌入，而torch.nn.Embedding就提供了这样的功能。
我们只需要初始化torch.nn.Embedding(n,m)，n是单词数，m就是词向量的维度。一开始embedding是随机的，在训练的时候会自动更新。
举个简单的例子：
word1和word2是两个长度为3的句子，保存的是单词所对应的词向量的索引号。
随机生成(4，5)维度大小的embedding，可以通过embedding.weight查看embedding的内容。输入word1时，embedding会输出第0、1、2行词向量的内容，word2同理。
import torch
word1 = torch.LongTensor([0, 1, 2])
word2 = torch.LongTensor([3, 1, 2])
embedding = torch.nn.Embedding(4, 5)
print(embedding.weight)
print('word1:')
print(embedding(word1))
print('word2:')
print(embedding(word2))

除此之外，我们也可以导入已经训练好的词向量，但是需要设置训练过程中不更新。
如下所示，emb是已经训练得到的词向量，先初始化等同大小的embedding，然后将emb的数据复制过来，最后一定要设置weight.requires_grad为False。
self.embedding = torch.nn.Embedding(emb.size(0), emb.size(1))
self.embedding.weight = torch.nn.Parameter(emb)
# 固定embedding
self.embedding.weight.requires_grad = False
"""


# Design model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)  # 相当于生成一个查询字典 (input_size,embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # (batch,seq_len,embedding_size)
        x, _ = self.rnn(x, hidden)  # input of rnn (batch_size,seq_len,embedding_size)
        x = self.fc(x)  # (batch, seq_len, hidden_size) ->(b,s, num_class)
        return x.view(-1, num_class)  # (bxs, num_class) reshape result to use crossEntropy


net = Model()

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# training cycle
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()  # 数组ndarray
    print('predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss=%.3f' % (epoch + 1, loss.item()))
