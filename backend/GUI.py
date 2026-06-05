import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import torch
import torch.nn.functional as F

# 导入你的项目模块
from data_process import get_dictionary, normalize_string
from model import HotelGRU
from backend.config.settings import CONFIG

# ===================== 全局配置与初始化 =====================
# matplotlib中文显示配置
matplotlib.use('Agg')
try:
    cn_font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=11)
    title_font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14, weight='bold')
except:
    try:
        cn_font = FontProperties(family="PingFang SC", size=11)
        title_font = FontProperties(family="PingFang SC", size=14, weight='bold')
    except:
        cn_font = FontProperties(family="WenQuanYi Micro Hei", size=11)
        title_font = FontProperties(family="WenQuanYi Micro Hei", size=14, weight='bold')

# 项目路径
data_file = r".\dataset\train.tsv"
model_path = r"./runs/model/model.pt"
log_path = r"./runs/logs/predict.txt"

# 加载词典与模型
x_y_pairs, word2index, index2word = get_dictionary(data_file)
vocab_size = len(word2index)
embedding_dim = CONFIG["embedding_dim"]
hidden_size = CONFIG["hidden_size"]
max_length = CONFIG["max_seq_len"]
device = CONFIG["device"]

model = HotelGRU(vocab_size, embedding_dim, hidden_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# ===================== 核心函数 =====================
def predict_sentiment():
    predict_btn.config(state=tk.DISABLED)
    status_label.config(text="正在预测中...", fg="orange")
    root.update()

    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        result_var.set("⚠️ 请输入要预测的酒店评论！")
        status_label.config(text="预测失败", fg="red")
        predict_btn.config(state=tk.NORMAL)
        return

    words = normalize_string(input_text)
    x = [word2index.get(word, 1) for word in words]
    x = x[:max_length] if len(x) > max_length else x + [0] * (max_length - len(x))
    tensor_x = torch.tensor(x, dtype=torch.long).to(device).unsqueeze(0)

    with torch.no_grad():
        output, attn_weights = model(tensor_x)
    pred_class = F.softmax(output, dim=1)
    pos_prob = pred_class[0][1].item()
    neg_prob = pred_class[0][0].item()
    sentiment = "正面好评" if pos_prob > neg_prob else "负面差评"

    result_text = f"【预测结果】{sentiment}\n正面概率：{pos_prob:.2%} | 负面概率：{neg_prob:.2%}"
    result_var.set(result_text)
    result_label.config(fg="green" if sentiment == "正面好评" else "red")

    update_charts(input_text, words, pos_prob, neg_prob, attn_weights)
    save_log(input_text, 1 if sentiment == "正面好评" else 0)
    update_history(input_text, sentiment, pos_prob, neg_prob)

    status_label.config(text="预测完成！", fg="green")
    predict_btn.config(state=tk.NORMAL)


def update_charts(input_text, words, pos_prob, neg_prob, attn_weights):
    # 只清空子图内容，不重新创建子图
    ax1.clear()
    ax2.clear()

    # --- 子图1：概率柱状图 ---
    sentiments = ['负面差评', '正面好评']
    probs = [neg_prob, pos_prob]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(sentiments, probs, color=colors, width=0.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('情感预测概率分布', fontproperties=title_font, pad=10)
    ax1.set_ylabel('概率', fontproperties=cn_font)

    # 永久隐藏y轴数值刻度
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # 柱子上标注概率值
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{prob:.2%}', ha='center', va='bottom', fontproperties=cn_font)

    # x轴分类标签
    ax1.set_xticks(range(len(sentiments)))
    ax1.set_xticklabels(sentiments, fontproperties=cn_font)

    # --- 子图2：注意力权重热力图（核心修复） ---
    # 1. 正确处理注意力权重维度
    attn_np = attn_weights.squeeze(0).cpu().numpy()  # 去掉batch维度 [50, 1]
    valid_len = min(len(words), max_length)
    valid_attn = attn_np[:valid_len, 0]  # 只取有效词, squeeze最后一维
    valid_attn_2d = valid_attn.reshape(1, -1)  # 转为2D数组，适配imshow

    # 2. 画热力图
    im = ax2.imshow(valid_attn_2d, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)  # 固定数值范围0-1
    ax2.set_title('模型注意力权重分布', fontproperties=title_font, pad=10)
    ax2.set_yticks([])  # 隐藏y轴

    # 3. 修复x轴标签：调整旋转角度、对齐方式，避免重叠
    ax2.set_xticks(range(valid_len))
    ax2.set_xticklabels(
        valid_words,
        fontproperties=cn_font,
        rotation=30,  # 调整为30度，避免过度旋转
        ha='right',
        fontsize=10
    )

    # 4. 热力图上标注权重值（只标0-1的正数）
    for i in range(valid_len):
        ax2.text(
            i, 0,
            f'{valid_attn[i]:.2f}',
            ha='center',
            va='center',
            color='black',
            fontproperties=cn_font,
            fontsize=9
        )

    # 5. 彻底解决颜色条重复：先清空所有旧颜色条，再加新的
    for cbar in fig.axes:
        if cbar != ax1 and cbar != ax2:
            cbar.remove()
    # 添加新颜色条，固定范围
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('注意力权重', fontproperties=cn_font)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # 固定刻度

    # 全局布局调整，避免挤压
    fig.suptitle(f'预测文本：{input_text}', fontproperties=title_font, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # 给顶部标题预留空间
    canvas.draw()


def save_log(text, label):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{text}\t{label}\n")


def update_history(text, sentiment, pos_prob, neg_prob):
    history_text.config(state=tk.NORMAL)
    history_text.insert(tk.END,
                        f"▶ 评论：{text}\n  结果：{sentiment} | 正面：{pos_prob:.2%} | 负面：{neg_prob:.2%}\n{'-' * 80}\n")
    history_text.config(state=tk.DISABLED)
    history_text.see(tk.END)


def clear_input():
    text_input.delete("1.0", tk.END)
    result_var.set("等待预测...")
    result_label.config(fg="black")
    status_label.config(text="就绪", fg="green")


def clear_history():
    history_text.config(state=tk.NORMAL)
    history_text.delete("1.0", tk.END)
    history_text.config(state=tk.DISABLED)


def on_closing():
    root.quit()
    root.destroy()


# ===================== 界面搭建 =====================
root = tk.Tk()
root.title("酒店评论情感分析系统")
root.geometry("1280x800")
root.protocol("WM_DELETE_WINDOW", on_closing)

style = ttk.Style()
style.configure("TButton", font=("SimHei", 11), padding=5)

# 标题区
title_frame = ttk.Frame(root, padding=10)
title_frame.pack(fill=tk.X)
title_label = tk.Label(title_frame, text="酒店评论智能情感分析系统", font=("SimHei", 20, "bold"))
title_label.pack()
status_label = tk.Label(title_frame, text="就绪", font=("SimHei", 10), fg="green")
status_label.pack()

# 输入区
input_frame = ttk.LabelFrame(root, text="评论输入", padding=10)
input_frame.pack(fill=tk.X, padx=20, pady=5)
text_input = scrolledtext.ScrolledText(input_frame, height=3, font=("SimHei", 12))
text_input.pack(fill=tk.X, pady=5)
btn_frame = ttk.Frame(input_frame)
btn_frame.pack(fill=tk.X, pady=5)
predict_btn = ttk.Button(btn_frame, text="开始预测", command=predict_sentiment)
predict_btn.pack(side=tk.LEFT, padx=5)
clear_btn = ttk.Button(btn_frame, text="清空输入", command=clear_input)
clear_btn.pack(side=tk.LEFT, padx=5)
clear_history_btn = ttk.Button(btn_frame, text="清空历史", command=clear_history)
clear_history_btn.pack(side=tk.LEFT, padx=5)
exit_btn = ttk.Button(btn_frame, text="退出系统", command=on_closing)
exit_btn.pack(side=tk.RIGHT, padx=5)

# 结果区
result_frame = ttk.LabelFrame(root, text="预测结果", padding=10)
result_frame.pack(fill=tk.X, padx=20, pady=5)
result_var = tk.StringVar(value="等待预测...")
result_label = tk.Label(result_frame, textvariable=result_var, font=("SimHei", 14, "bold"))
result_label.pack()

# 图表区（【关键修改4】这里一次性创建好两个子图对象）
chart_frame = ttk.LabelFrame(root, text="可视化分析", padding=10)
chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
fig = Figure(figsize=(12, 5), dpi=100)
# 【关键修改4】在初始化时就一次性创建好两个子图，之后只更新内容
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
canvas = FigureCanvasTkAgg(fig, master=chart_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 历史记录区
history_frame = ttk.LabelFrame(root, text="预测历史记录", padding=10)
history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
history_text = scrolledtext.ScrolledText(history_frame, height=4, font=("SimHei", 10), state=tk.DISABLED)
history_text.pack(fill=tk.BOTH, expand=True)

if __name__ == '__main__':
    root.mainloop()