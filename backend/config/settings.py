import os

# ===================== 项目根路径（自动计算） =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== 全局配置字典 =====================
CONFIG = {
    "PROJECT_ROOT": PROJECT_ROOT,
    # ---------------------- 数据路径 ----------------------
    "data_path": {
        "train": os.path.join(PROJECT_ROOT, "dataset", "train.tsv"),
        "val": os.path.join(PROJECT_ROOT, "dataset", "val.tsv"),
        "test": os.path.join(PROJECT_ROOT, "dataset", "test.tsv")
    },
    "save_dir": os.path.join(PROJECT_ROOT, "dataset", "saved_features"),
    "split_save_dir": os.path.join(PROJECT_ROOT, "dataset", "mask"),
    "log_dir": os.path.join(PROJECT_ROOT, "runs", "logs"),

    # ---------------------- 模型超参数 ----------------------
    "max_seq_len": 50,
    "embedding_dim": 128,
    "hidden_size": 256,
    "output_size": 2,
    "num_layers": 2,
    "dropout": 0.5,

    # ---------------------- 【新增】训练配置 ----------------------
    "model_type": "GRU",  # 切换模型：RNN / LSTM / GRU
    "learning_rate": 2e-4,  # 学习率
    "epochs": 8,  # 训练轮数
    "batch_size": 16,  # 批量大小
    "num_workers": 0,  # Windows 建议设为 0
    "device": "cuda",


    # ---------------------- 【新增】模型保存路径 ----------------------
    "model_save_dir": os.path.join(PROJECT_ROOT, "runs/model"),
    "model_path": os.path.join(PROJECT_ROOT, "runs/model", "model.pth")
}