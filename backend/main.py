from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F

from model import HotelGRU
from data_process import get_dictionary, normalize_string
from config.settings import CONFIG

# ===================== 初始化 =====================
app = FastAPI(title="酒店评论情感分析API")

# 允许跨域（前后端分离必须）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局加载模型（只加载一次，避免每次请求都加载）
device = CONFIG["device"]
data_file = "./dataset/train.tsv"
model_path = "./runs/model/model.pt"

x_y_pairs, word2index, index2word = get_dictionary(data_file)
vocab_size = len(word2index)
embedding_dim = CONFIG["embedding_dim"]
hidden_size = CONFIG["hidden_size"]
max_length = CONFIG["max_seq_len"]

model = HotelGRU(vocab_size, embedding_dim, hidden_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

# ===================== API定义 =====================


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    pos_prob: float
    neg_prob: float
    words: list
    attn_weights: list


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    input_text = request.text.strip()
    words = normalize_string(input_text)

    # 预处理
    x = [word2index.get(word, 1) for word in words]
    if len(x) > max_length:
        x = x[:max_length]
    else:
        x += [0] * (max_length - len(x))

    tensor_x = torch.tensor(x, dtype=torch.long).to(device).unsqueeze(0)

    # 推理
    with torch.no_grad():
        output, attn_weights = model(tensor_x)

    # 后处理
    pred_class = F.softmax(output, dim=1)
    pos_prob = float(pred_class[0][1].item())
    neg_prob = float(pred_class[0][0].item())
    sentiment = "正面好评" if pos_prob > neg_prob else "负面差评"

    # 只返回有效词的注意力权重（转成list，方便JSON传输）
    valid_len = min(len(words), max_length)
    valid_words = words[:valid_len]
    valid_attn = attn_weights[0][:valid_len].cpu().numpy().tolist()

    return PredictResponse(
        sentiment=sentiment,
        pos_prob=pos_prob,
        neg_prob=neg_prob,
        words=valid_words,
        attn_weights=valid_attn
    )


@app.get("/")
def read_root():
    return {
        "message": "酒店评论情感分析API服务已启动",
        "docs": "访问 http://127.0.0.1:8000/docs 查看API文档",
        "predict_endpoint": "POST http://127.0.0.1:8000/predict"
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)