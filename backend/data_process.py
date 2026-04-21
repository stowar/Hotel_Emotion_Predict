# -*- coding: utf-8 -*-
# 导入模块
import re
import jieba
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from backend.config.settings import CONFIG

# 测试
from model import HotelGRU

device = CONFIG["device"]
data_file = CONFIG["data_path"]["train"]

# todo:1.数据清洗
# 定义字符串清洗工具函数
def normalize_string(s):
    # 1.1 删除标点符号
    s = re.sub(r'[^\w\s]', '', s)
    # 1.2 进行结巴分词
    word_list = jieba.lcut(s)
    # 1.3 过滤停用词
    # 1.4 进行小写化
    return word_list

# todo:2.构建词典
# 读取文件,获取样本并且获取英文词典以及法文词典
def get_dictionary(filename):
    # 2.1 读取文件
    df = pd.read_csv(filename, sep='\t',header=None, escapechar=None)
    # print(df.head())

    # 2.2 获取样本对
    texts = df.iloc[:, 0].tolist()  # 第0列：评论文本
    labels = df.iloc[:, 1].tolist()  # 第1列：情感标签
    x_y_pairs = list(zip(texts, labels))  # 配对成 [(文本1, 标签1), (文本2, 标签2)...]

    # 2.3 获取清洗后的样本对
    x_y_pairs = [[normalize_string(x), y] for x, y in x_y_pairs]
    # print(x_y_pairs)

    # 2.4 获取词典
    # 2.4.1 获取word2index
    word2index = {"<PAD>": 0, "<UNK>": 1}
    word_n = 2
    for text, label in x_y_pairs:
        for word in text:
            if word not in word2index:
                word2index[word] = word_n
                word_n += 1
    # print(word2index)
    # 2.4.2 获取index2word
    index2word = {v: k for k, v in word2index.items()}

    return x_y_pairs, word2index, index2word


# 测试用:提前获取全局变量
x_y_pairs, word2index, index2word = get_dictionary(data_file)
# print(x_y_pairs)

# todo:3.构建Dataset数据源
class SeqDataset(Dataset):
    def __init__(self, x_y_pairs):
        super().__init__()
        # 获取样本对
        self.x_y_pairs = x_y_pairs
        # 获取样本总量
        self.sample_len = len(x_y_pairs)
        self.max_len = CONFIG["max_seq_len"]

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 异常值修正
        item = max(0, min(item, self.sample_len - 1))
        # 根据索引取出样本
        x = self.x_y_pairs[item][0]
        y = self.x_y_pairs[item][1]


        # x词转索引:找不到就返回
        x2index = [word2index.get(word, 1) for word in x]

        # 文本截断
        if len(x2index) > self.max_len:
            x2index = x2index[:self.max_len]
        # 文本填充
        x2index += [0] * (self.max_len - len(x2index))

        tensor_x = torch.tensor(x2index,dtype=torch.long,device=device)
        # 样本y 标签张量化表示
        tensor_y = torch.tensor(int(y),dtype=torch.long,device=device)
        return tensor_x, tensor_y

def test_dataset():
    dataset = SeqDataset(x_y_pairs)
    tensor_x, tensor_y = dataset[1]
    print(tensor_x)
    print(tensor_y)

# todo:4.实例化dataloader
def get_dataloader(x_y_pairs,batch_size,shuffle=True):
    seq_dataset = SeqDataset(x_y_pairs)
    seq_dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=shuffle)
    return seq_dataloader


if __name__ == '__main__':
    test_dataset()
    print(x_y_pairs)
    print(HotelGRU(vocab_size=len(word2index), embedding_dim=128, hidden_size=256))