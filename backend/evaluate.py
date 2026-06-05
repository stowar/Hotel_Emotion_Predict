# -*- coding: utf-8 -*-
# 导入模块
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# ===================== 导入项目自定义模块 =====================
from config.settings import CONFIG  # 全局配置（路径、超参数等）
from data_process import get_dictionary, get_dataloader  # 数据预处理流水线
from model import HotelGRU  # 三种情感分析模型

# 全局变量
tarin_data_file = CONFIG["data_path"]["train"]
test_data_file = CONFIG["data_path"]["test"]
train_x_y_pairs,train_word2index, train_index2word = get_dictionary(tarin_data_file)
test_x_y_pairs, test_word2index, test_index2word = get_dictionary(CONFIG["data_path"]["test"])

# 模型超参数
vocab_size = len(train_word2index)
embedding_dim = CONFIG["embedding_dim"]
hidden_size = CONFIG["hidden_size"]
max_length = CONFIG["max_seq_len"]

def evaluate():
    """
    模型评估核心逻辑
    :return: 平均损失、最终准确率、有效样本数
    """
    # 1.实例化测试集数据加载器
    dataloader = get_dataloader(test_x_y_pairs,batch_size=CONFIG["batch_size"],shuffle=False)

    # 2.加载模型
    model = HotelGRU(vocab_size, embedding_dim, hidden_size).to(CONFIG["device"])
    model.load_state_dict(torch.load(r".\runs\model\model.pt"))
    # 3.实例化优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    # 4.开始测试
    model.eval()
    # 4.1 定义测试日志
    total_loss = 0  # 总损失
    correct = 0  # 预测正确的样本数
    total = 0  # 总评估样本数
    total_samples = len(dataloader.dataset)  # 测试集总样本数
    # 评估阶段不计算梯度,提升速度、节省内存
    with torch.no_grad():
        for idx,(x,y) in enumerate(tqdm(dataloader,desc="评估中")):
            x = x.to(CONFIG["device"])
            y = y.to(CONFIG["device"])
            # 将x传入模型
            output, attn = model(x)
            # 前向传播
            loss = criterion(output,y) # 计算损失
            total_loss += loss.item() # 累加总损失
            pred = output.argmax(dim=1) # 取概率最大的为预测结果
            correct += (pred == y).sum().item() # 累加预测正确的样本数
            total += y.size(0) # 累加总评估样本数
            # 每100轮打印日志
            if idx % 100 == 0:
                current_acc = correct / total
                current_loss = total_loss / total
                print(f"📊 已评估 {idx + 1}/{total_samples} 条 | 当前准确率: {current_acc:.2%} | 当前损失: {current_loss:.4f}")

    # 计算最终评估指标
    avg_loss = total_loss / total if total > 0 else 0  # 平均损失
    final_acc = correct / total if total > 0 else 0  # 最终准确率
    # 保存最终日志
    with open(r".\runs\logs\evaluate.json", "w", encoding="utf-8") as f:
        log = {
            "avg_loss": avg_loss,
            "final_acc": final_acc,
            "total": total
        }
        json.dump(log, f, ensure_ascii=False, indent=4)
    return avg_loss, final_acc, total

if __name__ == '__main__':
    avg_loss, final_acc, total = evaluate()
    print(f"平均损失: {avg_loss:.4f} | 最终准确率: {final_acc:.2%} | 总评估样本数: {total}")

