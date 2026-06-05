# 导入库
import json

import torch
import torch.nn as nn
from data_process import get_dictionary, get_dataloader
from model import HotelGRU
import time
from config.settings import CONFIG
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 定义全局变量
# 配置路径
data_file = r".\dataset\train.tsv"

# 数据源加载
x_y_pairs, word2index, index2word = get_dictionary(data_file)

# 模型超参数
vocab_size = len(word2index)
embedding_dim = CONFIG["embedding_dim"]
hidden_size = CONFIG["hidden_size"]
max_length = CONFIG["max_seq_len"]

def train_model():
    # 1.构建dataloader
    dataloader = get_dataloader(x_y_pairs,batch_size=CONFIG["batch_size"])

    # 2.实例化模型
    model = HotelGRU(vocab_size, embedding_dim, hidden_size).to(CONFIG["device"])

    # 3.实例化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # 4.开始训练
    start_time = time.time()
    # 定义全局绘图列表（只在epoch结束时添加平均loss/acc，避免列表太长）
    epoch_loss_list = []
    epoch_acc_list = []
    epoch_attn_list = []
    # 开始外部遍历
    for epoch in range(CONFIG["epochs"]):
        # 定义训练日志
        epoch_total_loss = 0.0
        epoch_total_correct = 0
        epoch_total_samples = 0
        iter_loss_list = []  # 用于绘图的单轮iter loss
        model.train()
        print(f"Epoch: {epoch+1}/{CONFIG['epochs']}")
        for idx,(x, y) in enumerate(tqdm(dataloader,desc=f"第{epoch+1}轮训练")):
            x = x.to(CONFIG["device"])
            y = y.to(CONFIG["device"])
            # 将x传入模型
            output, attn = model(x)
            # 前向传播
            loss = criterion(output, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            iter_loss_list.append(loss.item())
            pred = output.argmax(dim=1)
            epoch_total_correct += (pred == y).sum().item()
            epoch_total_samples += y.size(0)
            # print(f"第{total_iter_num}次迭代，总损失：{total_loss:.4f}，总准确率：{total_acc/total_iter_num:.4f}")
        # 计算单轮的平均loss和准确率
        epoch_avg_loss = epoch_total_loss / len(dataloader)
        epoch_avg_acc = epoch_total_correct / epoch_total_samples  # 分母是总样本数！
        epoch_loss_list.append(epoch_avg_loss)
        epoch_acc_list.append(epoch_avg_acc)
        epoch_attn_list.append(attn.tolist())
        # 打印单轮结果
        print(f"第{epoch + 1}轮训练结束，平均损失：{epoch_avg_loss:.4f}，总准确率：{epoch_avg_acc:.4%}")
        print(f"第{epoch + 1}轮训练结束，总耗时：{time.time() - start_time:.4f}s")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(CONFIG["model_save_dir"], "model.pt"))
    print(f"模型保存成功，保存路径：{os.path.join(CONFIG['model_save_dir'], 'model.pt')}")

    # 保存日志
    json.dump(epoch_loss_list, open(os.path.join(CONFIG["log_dir"], "epoch_loss_list.json"), "w"))
    json.dump(epoch_acc_list, open(os.path.join(CONFIG["log_dir"], "epoch_acc_list.json"), "w"))
    json.dump(epoch_attn_list, open(os.path.join(CONFIG["log_dir"], "epoch_attn_list.json"), "w"))

    plt.figure(0)
    plt.title("HotelGRU_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1, CONFIG["epochs"]+1), epoch_loss_list, marker='o')
    plt.savefig("./runs/logs/HotelGRU_loss.png")
    plt.show()

    # 画准确率曲线
    plt.figure(1)
    plt.title("HotelGRU_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.plot(range(1, CONFIG["epochs"]+1), epoch_acc_list, marker='s', color='orange')
    plt.savefig("./runs/logs/HotelGRU_acc.png")
    plt.show()

    # 画注意力曲线（取最后一个batch的第一个样本，形状[50,1]→[50]）
    plt.matshow(attn[0].detach().cpu().numpy().squeeze(-1).reshape(1, -1))
    plt.savefig("./runs/logs/HotelGRU_attn.png")
    plt.show()


if __name__ == '__main__':
    train_model()