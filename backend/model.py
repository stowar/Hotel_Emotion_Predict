# -*- coding: utf-8 -*-
# ===================== 1. 导入依赖库 =====================
import torch
import torch.nn as nn
from backend.config.settings import CONFIG

class Selfattention(nn.Module):
    def __init__(self, hidden_size):
        super(Selfattention, self).__init__()
        self.hidden_size = hidden_size
        # 定义全连接层：把隐藏层特征映射为1个注意力分数
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, gru_output):
        """
        前向传播
        :param gru_output: GRU的输出 [批次大小, 句子长度, 隐藏层维度] → [32, 50, 256]
        :return: 加权聚合后的特征 [批次大小, 隐藏层维度] → [32, 256]
        """
        # 1. 计算每个词的注意力分数
        attn_scores = self.attention(gru_output)
        # 2. softmax归一化权重(和为1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # 3. 加权求和：用注意力权重乘以对应词的特征，再求和
        context_vector = torch.sum(gru_output * attn_weights, dim=1)
        return context_vector

class HotelGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout'], pad_idx=0):
        super(HotelGRU, self).__init__()
        """
        酒店情感分类主模型：Embedding + GRU + 自注意力 + 全连接层
        传参说明（必须记）：
        vocab_size:     词典大小（自动从数据中获取）
        embedding_dim:  词向量维度 (128)
        hidden_size:    GRU隐藏层维度 (256)
        num_layers:     GRU层数 (2)
        dropout:        失活率 (0.2)
        pad_idx:        填充符<PAD>的索引 (0)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 1.定义词嵌入层:把词索引转为向量 2维->3维
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,    # 词典大小
            embedding_dim=embedding_dim,  # 词向量维度
            padding_idx=pad_idx           # 指定填充符<PAD>，不参与训练
        )
        # 2.定义GRU层
        self.gru = nn.GRU(
            input_size=embedding_dim,  # 输入维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # 3.定义自注意力层
        self.attention = Selfattention(hidden_size)
        # 4.定义dropout层:防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 5.定义全连接层:分类输出
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self,text):
        """
        前向传播
        :param text: 输入文本 [批次大小, 句子长度] → [1, 50]
        :return: 分类结果 [批次大小, 类别数] → [1, 2]
        """
        # 1.词嵌入->[1,50,128]
        embedded = self.embedding(text)
        # 2.GRU提取 -> output,hidden -> output: [1,50,256], hidden: [2,1,256]
        gru_output, hn = self.gru(embedded)
        # 3.自注意力层加权 -> [1,256]
        attn_attention = self.attention(gru_output)
        # 4.dropout
        dropout_output = self.dropout(attn_attention)
        # 5.全连接层 -> [1,2]
        output = self.fc(dropout_output)
        return output,attn_attention






