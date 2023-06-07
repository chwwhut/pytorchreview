"""
根据名字识别他所在的国家
人名字符长短不一，最长的10个字符，所以处理成10维输入张量，都是英文字母刚好可以映射到ASCII上
Maclean ->  ['M', 'a', 'c', 'l', 'e', 'a', 'n'] ->  [ 77 97 99 108 101 97 110]  ->  [ 77 97 99 108 101 97 110 0 0 0]
共有18个国家，设置索引为0-17
训练集和测试集的表格文件都是第一列人名，第二列国家
"""
import math
import torch
import time
import csv
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = True


# preparing data
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv' if is_train_set else 'names_test.csv'
        with open(filename) as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]  # save names
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]  # save countries and its index in list and dictionary
        self.country_list = list(sorted(set(self.countries)))  # 国家名集合，18个国家名的集合 0-17
        self.country_dict = self.getCountryDict()  # 转变成词典 18个国家名 0-17
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]  # key-values

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]  # return country name giving index

    def getCountriesNum(self):
        return self.country_num


# prepare dataset and dataloader
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.country_num  # N_COUNTRY is the output size of our model


def create_tensor(tensor):  # 判断是否使用GPU 使用的话把tensor搬到GPU上去
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


# Model design
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size  # parameters of GRU layer
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # input of emb shape (seq_len ,batchSize) -> output( seq_len ,batchSize, hidden_size)
        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # input_size ? N_CHARS=128 ascii码值个数
        # inputs of GRU ( seq_len ,batchSize, hidden_size)  hidden (n_layers*n_directions, batchSize,hiddenSize)
        # outputs ( seq_len ,batchSize, hidden_size*n_directions)  hidden (n_layers*n_directions, batchSize,hiddenSize)
        # n_direction = 2  [h(f)n, h(b)n]拼接
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)
        # (seq_len ,batchSize, hidden_size*n_directions) -> (seq_len ,batchSize, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        # input shape : BXS -> S->B
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)  # 初始化隐藏层 (n_layers*n_directions, batchSize,hiddenSize)
        embedding = self.embedding(input)  # result of emb (seq_len ,batchSize, hidden_size)

        # pack_padded_sequence函数当输入seq_lengths是GPU张量时报错，在这里改成cpu张量就可以，不用GPU直接注释掉下面这一行代码
        seq_lengths = seq_lengths.cpu()  # 改成cpu张量

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 让0值不参与运算加快运算速度的方式
        # seq_length: a list of sequence length of each batch element
        # 需要提前把输入按有效值长度降序排列 再对输入做嵌入，然后按每个输入len（seq——lengths）取值做为GRU输入

        output, hidden = self.gru(gru_input, hidden)  # 双向传播的话hidden有两个
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        """
        如果该GRU层是双向的（n_directions为2），则将前向和后向的最终隐藏状态拼接起来，形成一个新的向量hidden_cat，
        其中hidden[-1]表示最后一个时间步的前向隐藏状态，hidden[-2]表示最后一个时间步的后向隐藏状态；
        如果该GRU层是单向的（n_directions为1），则仅使用最后一个时间步的隐藏状态作为hidden_cat。
        """
        fc_output = self.fc(hidden_cat)
        return fc_output  # (seq_len ,batchSize, output_size)


# 对名字的处理需要先把每个名字按字符都变成ASCII码
def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)  # 输出元组


# convert name to tensor
def make_tensors(names, countries):  # # 处理名字ASCII码 重新排序的长度和国家列表
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # 用名字列表的ASCII码填充上面的全0tensor

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # 将seq_lengths按序列长度重新降序排序，返回排序结果和排序序列。
    # 说明：m.sort(dim = _, descending = _),dim=0或1，0是按列排序，1是按行排序；
    # descending=True是由大到小，false是由小到大。最后返回：1.排序后的序列；2.排序后对应的原来的idx。
    # 将这两个也按照相同的idx排序
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# one epoch training
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)  # 取出排序后的 ASCII列表 名字长度列表 国家名列表
        output = classifier(inputs, seq_lengths)  # 把输入和长度序列放入分类器
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 打印输出结果
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]   # ? dim = 1, SXB,output_size?
            """
            torch.max的用法
            (max, max_indices) = torch.max(input, dim, keepdim=False)
            输入：
            1、input 是输入的tensor。
            2、dim 是索引的维度，dim=0寻找每一列的最大值，dim=1寻找每一行的最大值。
            3、keepdim 表示是否需要保持输出的维度与输入一样，keepdim=True表示输出和输入的维度一样，keepdim=False表示输出的维度被压缩了，
               也就是输出会比输入低一个维度。
            输出：
            1、max 表示取最大值后的结果。
            2、max_indices 表示最大值的索引。
            """
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算预测对了多少
            # pred.eq() 是 PyTorch 中的一个函数，用于计算两个张量的逐元素相等比较结果。

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda:0')
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # 计算损失
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 更新

    start = time.time()
    print("Train for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # 训练
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

# torch.nn.utils.rnn.pack_padded_sequence是PyTorch中的一个函数，
# 用于将一个填充过的序列打包成一个紧凑的Tensor。这个函数通常用于处理变长的序列数据，
# 例如自然语言处理中的句子。打包后的Tensor可以传递给RNN模型进行训练或推理，以提高计算效率和减少内存占用。
