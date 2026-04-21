
# 酒店评论情感分析系统
> 基于 PyTorch + FastAPI 实现的中文酒店评论情感分析服务，支持单条文本预测、批量文件处理、注意力权重可视化，开箱即用，CPU/GPU双环境均可稳定运行。

---

## 一、项目简介
本项目是**端到端的酒店评论情感分析解决方案**，核心目标是自动化识别酒店用户评论的情感倾向（正面好评/负面差评），同时通过注意力机制输出模型决策的关注重点，配套完整的Web API服务与可视化能力，可直接用于酒店用户反馈分析、口碑监控等业务场景。

### 核心功能
- ✅ 单条评论情感预测：返回情感标签、正负向概率、分词结果与注意力权重
- ✅ 批量评论处理：支持CSV/TSV格式文件的批量预测与结果导出
- ✅ 可视化能力：输出注意力权重数据，可直接对接前端生成热力图，直观展示模型决策依据
- ✅ 跨域兼容：原生支持前后端分离架构，前端可直接调用接口
- ✅ 双环境兼容：原生支持CPU/GPU环境，无需额外修改代码即可切换

---

## 二、技术栈
| 模块 | 技术选型 | 推荐稳定版本 |
| :--- | :--- | :--- |
| 后端Web框架 | FastAPI + Uvicorn | fastapi==0.104.1，uvicorn==0.24.0 |
| 深度学习框架 | PyTorch | 2.2.0（CPU版/CUDA 11.7版双版本兼容） |
| 模型架构 | 双向GRU + Attention注意力机制 | 自定义实现，支持权重可视化 |
| 中文NLP处理 | jieba | 0.42.1（中文分词与文本预处理） |
| 数据处理 | Pandas + NumPy | pandas==2.1.4，numpy==1.26.4 |
| 数据校验 | Pydantic | 2.4.2（FastAPI原生依赖） |
| 可视化 | ECharts | 前端适配，后端输出标准化权重数据 |
| Excel文件支持 | openpyxl | 最新稳定版 |

---

## 三、项目结构
```
Hotel_Emotion(refactor)_2.0/
├── backend/                          # 后端核心目录
│   ├── main.py                       # FastAPI服务入口，API接口定义
│   ├── model.py                      # 模型定义：HotelGRU + Attention层
│   ├── data_process.py               # 数据预处理工具：分词、词典构建、文本标准化
│   ├── predict.py                    # 独立推理脚本，可本地批量测试
│   └── config/
│       └── settings.py               # 全局配置：超参数、设备、路径配置
├── dataset/                          # 数据集目录
│   └── train.tsv                     # 训练数据集，TSV格式（文本+标签）
├── runs/
│   └── model/
│       └── model.pt                  # 训练完成的模型权重文件
├── frontend/                         # 前端可视化页面
│   └── index.html                    # 前端可视化页面，直接打开即可对接后端接口
├── requirements.txt                  # 项目基础依赖清单
└── README.md                         # 项目说明文档
```

---

## 四、环境准备
### 基础环境要求
- Python 版本：3.8 ~ 3.12（推荐3.12，已全量验证兼容）
- 运行环境：Windows/Linux/MacOS 均可，**纯CPU环境即可稳定运行**，有NVIDIA显卡可启用GPU加速
- 内存要求：最低2GB可用内存，推荐4GB以上
- GPU环境要求（可选）：NVIDIA显卡，CUDA 11.7版本，驱动版本≥450.80.02

---

## 五、安装与部署
### 1. 克隆/下载项目
将项目文件下载到本地，进入项目根目录
```bash
cd Hotel_Emotion(refactor)_2.0
```

### 2. （推荐）创建虚拟环境
隔离项目依赖，避免与系统Python环境冲突
#### 方式一：venv虚拟环境（Python自带，无需额外安装）
```bash
# 创建虚拟环境
python -m venv venv
# Windows系统激活
venv\Scripts\activate
# Linux/MacOS系统激活
source venv/bin/activate
```

#### 方式二：Anaconda虚拟环境（推荐GPU用户使用）
```bash
# 使用Anaconda创建虚拟环境
conda create -n hotel_emotion python=3.12 -y
conda activate hotel_emotion
```

### 3. 安装依赖
> 若清华镜像源访问失败，可将命令中的`https://pypi.tuna.tsinghua.edu.cn/simple`替换为备用镜像：
> - 阿里云镜像：`https://mirrors.aliyun.com/pypi/simple/`
> - 中科大镜像：`https://pypi.mirrors.ustc.edu.cn/simple/`

#### 版本一：CUDA GPU版（有NVIDIA显卡推荐）
```bash
# 1. 安装基础依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 2. 单独安装PyTorch GPU版（CUDA 11.7）
pip install torch==2.2.0+cu117 torchvision==0.17.0+cu117 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu117
```

#### 版本二：CPU版（无显卡最简版本）
若安装CUDA版失败，或无NVIDIA显卡，直接安装PyTorch CPU版
```bash
# 1. 安装基础依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 2. 单独安装PyTorch CPU稳定版
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## 六、快速启动
### 1. 启动后端API服务
```bash
# 进入后端目录
cd backend
# 启动服务
python main.py
```

### 2. 验证服务状态
启动成功后，终端会输出如下日志：
```
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
> 注意：Windows系统请使用`http://127.0.0.1:8000`访问，`0.0.0.0`在Windows下无法直接访问
- 服务根地址：`http://127.0.0.1:8000`
- 在线API文档（可在线测试）：`http://127.0.0.1:8000/docs`
- 核心预测接口：`http://127.0.0.1:8000/predict`
- 前端页面：直接打开`frontend/index.html`即可使用可视化界面

---

## 七、API接口文档
### 1. 服务状态检查
- 请求方式：`GET /`
- 接口说明：检查服务是否正常启动，返回服务基础信息
- 响应示例：
```json
{
  "message": "酒店评论情感分析API服务已启动",
  "docs": "访问 http://127.0.0.1:8000/docs 查看API文档",
  "predict_endpoint": "POST http://127.0.0.1:8000/predict"
}
```

### 2. 单条评论情感预测
- 请求方式：`POST /predict`
- 接口说明：核心预测接口，输入单条评论文本，返回情感分析结果与注意力权重
- 请求体（JSON格式）：
```json
{
  "text": "酒店位置很好找，房间干净整洁，前台服务态度也特别贴心"
}
```
- 响应示例：
```json
{
  "sentiment": "正面好评",
  "pos_prob": 0.9976,
  "neg_prob": 0.0024,
  "words": ["酒店", "位置", "很好找", "房间", "干净", "整洁", "前台", "服务", "态度", "特别", "贴心"],
  "attn_weights": [0.08, 0.10, 0.22, 0.07, 0.15, 0.12, 0.05, 0.06, 0.04, 0.03, 0.08]
}
```
- 字段说明：
  | 字段名 | 含义 |
  | :--- | :--- |
  | sentiment | 最终情感标签，正面好评/负面差评 |
  | pos_prob | 正面情感概率，范围0~1 |
  | neg_prob | 负面情感概率，范围0~1 |
  | words | 分词后的有效词语列表 |
  | attn_weights | 对应每个词语的注意力权重，权重越高，模型对该词的关注度越高 |

---

## 八、模型说明
### 核心架构
本项目采用**双向GRU + Attention**的轻量级序列模型，兼顾推理速度与语义捕捉能力，同时适配CPU/GPU双环境：
1.  **嵌入层（Embedding）**：将分词后的词语映射为固定维度的稠密向量，实现文本的向量化表示
2.  **双向GRU层**：双向捕捉文本的上下文语义特征，解决单向RNN的长距离依赖问题
3.  **注意力层（Attention）**：自动学习每个词语对情感分类的贡献权重，输出可解释的决策依据，支持可视化
4.  **全连接输出层**：输出二分类的情感概率，通过softmax归一化得到最终正负向概率

### 推理流程
1.  文本标准化：去除特殊字符、标点符号，使用jieba完成中文分词
2.  序列处理：将分词结果映射为词典索引，完成序列的截断/补零，统一为固定长度
3.  模型推理：输入张量进入模型，关闭梯度计算，自动适配CPU/GPU设备，输出分类logits与注意力权重
4.  结果后处理：通过softmax得到概率分布，判断最终情感标签，过滤无效权重，输出标准化结果

---

## 九、完整依赖清单（requirements.txt）
```txt
# ==================== 核心后端框架 ====================
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# ==================== 中文NLP与数据处理 ====================
jieba==0.42.1
numpy==1.26.4
pandas==2.1.4
matplotlib==3.8.2
tqdm==4.66.1
openpyxl

# ==================== 可选：PyTorch安装请参考文档执行单独命令 ====================
# torch==2.2.0
# torchvision==0.17.0
# torchaudio==2.2.0
```

---

## 十、常见问题与解决方案
### 1. 运行报错：`SDK is not defined for Run Configuration`
- 问题原因：PyCharm运行配置未绑定Python解释器
- 解决方案：
  1.  右键`main.py`空白处，选择`Run 'main'`，IDE会自动生成正确配置
  2.  手动配置：右上角运行配置 → `Edit Configurations` → 选择Python解释器 → 保存后重新运行

### 2. 模型加载报错：`RuntimeError: Attempting to deserialize object on a CUDA device`
- 问题原因：模型权重是GPU环境训练保存的，当前环境无CUDA，加载失败
- 解决方案：修改`main.py`中的模型加载代码，强制映射到CPU
```python
# 错误代码
model.load_state_dict(torch.load(model_path, map_location=device))
# 修正代码
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to("cpu")
```

### 3. 导包报错：`ModuleNotFoundError: No module named 'backend'`
- 问题原因：使用了错误的绝对导包路径，运行时无法识别`backend`包
- 解决方案：修改同级目录的导包语句，去掉`backend.`前缀
```python
# 错误代码
from backend.config.settings import CONFIG
# 修正代码
from config.settings import CONFIG
```

### 4. 依赖安装报错：`Could not find a version that satisfies the requirement`
- 问题原因：国内网络无法连接PyPI源，或Python版本与依赖版本不兼容
- 解决方案：更换国内镜像源安装，或手动指定与Python版本兼容的依赖版本

### 5. Excel读取报错：`Import openpyxl failed`
- 问题原因：Pandas读取Excel文件需要`openpyxl`依赖，未安装
- 解决方案：执行`pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 6. 推理报错：`AttributeError: 'tuple' object has no attribute 'softmax'`
- 问题原因：模型返回的是`(output, attention_weights)`元组，不是单纯的tensor
- 解决方案：修改推理代码，只取第一个输出值
```python
# 错误代码
output = model(input_tensor)
pred_class = F.softmax(output, dim=1)
# 修正代码
output, _ = model(input_tensor)
pred_class = F.softmax(output, dim=1)
```

### 7. 服务启动后，页面/接口访问报错：`URL拼写可能存在错误`/无法访问
- 问题原因：Windows系统地址兼容问题、端口占用、防火墙拦截、服务未正常启动
- 解决方案（按顺序排查）：
  1.  确认终端无报错，服务正常启动，使用`http://127.0.0.1:8000`访问，不要用`0.0.0.0`
  2.  检查8000端口是否被占用：Windows执行`netstat -ano | findstr "8000"`，Linux执行`lsof -i:8000`，关闭占用端口的程序，或修改启动端口
  3.  关闭Windows防火墙/系统代理，或添加8000端口的防火墙入站规则
  4.  确认uvicorn版本为0.24.0，执行`pip install uvicorn==0.24.0`重新安装

---

## 十一、扩展与优化方向
1.  **数据集优化**：补充更多酒店场景的真实评论，尤其是反讽、隐晦评价、中性偏负面等难分样本，提升模型鲁棒性
2.  **模型升级**：替换为中文预训练模型（如BERT、RoBERTa），大幅提升复杂语义的理解能力
3.  **功能扩展**：新增批量文件上传接口、评论多标签分类（如服务、卫生、位置等维度）、口碑统计报表生成
4.  **部署优化**：提供Docker容器化部署方案，支持云服务器一键部署、Nginx反向代理配置
5.  **前端优化**：完善可视化界面，新增注意力热力图、批量结果筛选、数据统计图表等功能

---

## 十二、声明
本项目仅供学习与研究使用，请勿直接用于商业用途。

---
---
### 项目开发心路（放在更新日志前，真实还原边学边做的成长过程）
本项目为个人大一阶段的AI入门全栈实践项目，全程**边学边做、以练促学**：立项时仅掌握Python与PyTorch基础，对注意力机制、前后端分离架构、FastAPI服务开发均无系统认知，所有功能与架构均是随学习进度逐步迭代、重构完善，完整记录了从0到1落地AI项目的全成长过程。

---

---
### 项目开发心路（放在更新日志前）
本项目为个人大一阶段的**AI入门全栈练手项目**，全程**边查边做、以练促学、不刻意追求完美规范**：
- 立项时仅掌握Python基础语法与PyTorch入门张量操作，对注意力机制、前后端分离、FastAPI/ECharts等技术均无系统认知；
- 所有功能、架构均是随项目需求逐步迭代、遇到问题查资料解决后完善的；
- 完整记录了从“只会写简单Python脚本”到“能落地一个带界面的AI小工具”的全成长过程。

---

## 十三、更新日志
| 版本号 | 发布日期       | 版本类型          | 核心更新内容（完全真实对应学习/解决问题节点）                                                                                                                                                                                                 |
|--------|--------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v2.1.0 | 2026-04-21   | 最终可用稳定版    | 1. 完成基础README.md文档编写，补充环境兼容说明、常见报错（如CPU加载、导包）的临时解决方案<br>2. 修复模型CPU加载报错、导包路径错误<br>3. 整理requirements.txt依赖清单，提供国内多镜像源安装提示                                                                                                       |
| v1.2.0 | 2026-04-21   | 模型效果优化版    | 1. 修复 `tuple has no attribute softmax` 核心推理报错<br>2. 手动新增反讽、隐晦语义测试样本，调整部分文本预处理逻辑，优化对这类复杂评论的识别能力                                                                                          |
| v2.0.0 | 2026-04-20   | 前后端雏形完善版  | 1. 通过FastAPI官方文档、教程快速了解基础用法，尝试搭建前后端分离雏形并逐步完善<br>2. 用HTML/CSS/JS写简单前端可视化交互界面，支持单条预测、批量预测与结果筛选<br>3. 跟着ECharts官方示例快速上手，实现基础的注意力权重热力图、情感概率分布柱状图<br>4. 查资料解决CORS跨域问题，加Swagger在线调试文档方便测试<br>5. 优化模型加载逻辑，全局仅加载一次，大幅提升接口响应速度<br>6. 新增Excel/CSV批量文件预测与结果导出功能 |
| v1.1.0 | 2026-04-18   | 环境兼容适配版    | 1. 查资料解决CUDA训练模型在CPU环境加载失败的核心问题<br>2. 修复绝对导包路径错误，完成无GPU纯CPU运行环境适配<br>3. 补充PyCharm SDK未配置、依赖安装失败的临时解决方案<br>4. 整理国内镜像源安装方式，解决依赖网络安装慢的问题                     |
| v1.0.1 | 2026-04-17   | 注意力机制尝试版  | 1. 发现基础GRU对长评论效果不好，针对性学习Attention注意力机制核心逻辑，将原模型调整为「Embedding+单层双向GRU+Attention」架构<br>2. 重写全链路代码逻辑，包括数据读取、词典构建、模型训练、保存、加载与推理<br>3. 解决序列长度不统一导致的推理报错问题<br>4. 新增注意力权重输出，能大概看到模型关注哪些词<br>5. **测试集准确率从0.75提升至0.86，模型泛化能力明显增强**                                                              |
| v1.0.0 | 2026-03-21   | 基础模型本地版    | 1. 跟着教程学习RNN/GRU序列模型基础用法，搭建双向GRU基础情感分类模型<br>2. 实现中文分词、文本归一化、词典构建全流程数据预处理<br>3. 完成模型训练、权重保存、模型加载核心逻辑开发<br>4. 实现本地控制台单条评论情感基础预测功能<br>5. **基础模型测试集准确率0.75**                                                                 |
| v0.9.0 | 2026-03-16   | 数据集与基础框架版 | 1. 学习NLP数据预处理基础方法，收集酒店评论公开数据集，完成正负标签清洗、去重与简单整理<br>2. 跟着教程设计初始模型架构：Embedding+基础双向GRU<br>3. 封装数据读取、清洗、归一化的简单工具函数<br>4. 搭建项目基础文件夹结构，确定项目核心方向                      |
| v0.1.0 | 2026-03-15   | 项目启动立项版    | 1. 正式确定项目主题：酒店评论智能情感分析系统，作为个人AI入门练手项目<br>2. 完成基础技术选型：以PyTorch为核心框架，目标先实现本地控制台的基础文本情感二分类<br>3. 搭建项目基础文件夹结构<br>4. 开始学习Python基础语法、PyTorch入门与数据预处理相关知识                                                                             |

---
给你准备了3个不同风格的结项说明，直接选一个复制到README末尾即可，和更新日志放在一起，上传GitHub就正式闭环了。

---
## 项目结项说明
本项目已于 **2026年04月21日** 完成全部核心功能开发与最终稳定版(v2.1.0)发布，正式结项。

项目完整覆盖从需求立项、数据预处理、模型开发训练、前后端工程化落地、文档完善的全生命周期，所有核心功能均已完成开发与测试，可正常开箱使用。

后续本仓库将以维护为主，仅修复兼容性问题与严重BUG，不再新增核心功能；进阶迭代内容将在新分支/新仓库中开展。
---


---