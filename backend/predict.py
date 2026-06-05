# 导入库
import torch
from data_process import get_dictionary
from model import HotelGRU
from data_process import normalize_string
from config.settings import CONFIG
import torch.nn.functional as F

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
device = CONFIG["device"]

def model_predict():
    # 初始化模型
    model = HotelGRU(vocab_size, embedding_dim, hidden_size)
    model.load_state_dict(torch.load(r"./runs/model/model.pt", map_location=torch.device('cpu')))
    model.eval()
    while True:
        text = input("请输入要预测的文本：")
        if text == "0":
            break
        # 文本预处理
        words = normalize_string(text)
        x = [word2index.get(word, 1) for word in words]

        # 文本截断
        if len(x) > max_length:
            x = x[:max_length]
        # 文本填充
        x += [0] * (max_length - len(x))

        tensor_x = torch.tensor(x, dtype=torch.long).to(torch.device('cpu'))
        # print(tensor_x.shape)
        tensor_x = tensor_x.unsqueeze(0)
        # print(tensor_x.shape)

        with torch.no_grad():
            output,_ = model(tensor_x)

        pred_class = F.softmax(output, dim=1) # 单元素标量
        sentiment = "正面评论" if pred_class[0][1] > pred_class[0][0] else "负面评论"
        print(f"预测结果:\n{sentiment}|正面概率：{pred_class[0][1]:.2%}|负面概率：{pred_class[0][0]:.2%}")
        with open(r"./runs/logs/predict.txt", "a", encoding="utf-8", errors="ignore") as f:
            f.write(f"{text}\t{1 if sentiment == '正面评论' else 0}\n")


if __name__ == '__main__':
    model_predict()



