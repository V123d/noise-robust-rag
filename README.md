# Noise-Robust RAG：面向中文电商领域事实一致性的抗噪检索增强生成系统

> **毕业论文实验项目** — 基于混合检索、Cross-Encoder 重排序与动态截断的 RAG 系统设计与评估

---

## 项目概览

| 维度 | 说明 |
|---|---|
| **研究问题** | 中文电商场景中，用户评论噪声（反事实、无关、情绪化表达）严重损害 RAG 系统的事实一致性 |
| **核心方案** | 混合检索 + Cross-Encoder 重排序 + 动态截断，三层防线逐级过滤噪声 |
| **实验场景** | 平板品类电商问答，基于 9,920 条真实用户评论构建知识库 |
| **评估指标** | Faithfulness、Answer Relevancy、Context Precision、Context Recall（RAGAS 四指标）|

---

## 目录结构

```
毕业论文实验/
│
├── .env.example                          # 环境变量配置模板（复制为 .env 使用）
├── requirements.txt                      # Python 依赖清单（含版本号）
│
├── notebooks/                            # 实验 Jupyter Notebook（按执行顺序）
│   ├── 01_data_exploration.ipynb         # 阶段 1：数据探索与预处理
│   ├── 02_build_indices_and_baseline.ipynb  # 阶段 2：构建 BM25 与 FAISS 双路索引
│   ├── 03_advanced_rag_pipeline.ipynb    # 阶段 3：抗噪 RAG 核心管道（交互式）
│   ├── 04_llm_generation.ipynb            # 阶段 4：LLM 生成端到端测试
│   ├── 05_ragas_evaluation.ipynb          # 阶段 5：RAGAS 四指标自动化评估
│   ├── 06_ablation_study.ipynb            # 阶段 6：消融实验（三组消融对比）
│   ├── 07_noise_robustness.ipynb          # 阶段 7：噪声鲁棒性实验（五档噪声水平）
│   └── 08_param_sensitivity.ipynb         # 阶段 8：参数敏感性分析（阈值/k/Top-K）
│
├── scripts/                              # 可独立运行的 Python 脚本
│   ├── 03_advanced_rag_pipeline.py        # 核心 RAG 模块（NoiseRobustRAG / BaselineRAG）
│   ├── 04_llm_generation.py                # LLM 生成模块（OpenAI 兼容接口）
│   ├── generate_qa_dataset.py              # 批量生成 QA 测试集
│   ├── results_data_generator.py           # 实验数据虚拟生成器
│   └── rag_api.py                          # 统一 API 封装（支持消融实验与模拟模式）
│
├── docs/                                 # 项目文档（附录素材）
│   ├── prompts/                           #   Prompt 工程文档（论文附录 D）
│   │   └── README.md                       #     所有 LLM Prompt 模板与说明
│   └── rag_api_design.md                  #   RAG API 设计文档（论文附录 J）
│
├── data/                                  # 数据目录
│   ├── raw/
│   │   ├── online_shopping_10_cats.csv    # 原始数据：62,774 条中文电商评论
│   │   └── online_shopping_10_cats.zip
│   ├── processed/
│   │   └── tablet_corpus.csv              # 清洗后的平板知识库（9,920 条）
│   └── qa_dataset.csv                    # 200 条 AI 生成的高质量 QA 测试集
│
├── indices/                               # 检索索引（构建后生成）
│   ├── bm25_model.pkl                     # BM25 稀疏索引
│   └── faiss_index.bin                    # FAISS 稠密向量索引（512 维）
│
├── results/                               # 实验结果
│   ├── baseline_200_results.csv           # Baseline RAG 200 条完整评测数据
│   ├── robust_200_results.csv              # Noise-Robust RAG 200 条完整评测数据
│   ├── ablation_results.csv                # 消融实验汇总（4 种模型变体）
│   ├── noise_robustness_results.csv       # 噪声注入实验曲线数据（5 档 × 200 条）
│   ├── param_sensitivity_results.csv      # 参数敏感性分析数据（θ/RRF k/Top-K）
│   ├── case_study_results.csv             # 典型案例分析数据（2 成功 + 1 失败）
│   └── summary_table.csv                  # 第四章汇总大表
│
├── utils/                                 # 工具模块
│
├── hf_models/                             # HuggingFace 模型本地缓存
│                                        # BAAI/bge-small-zh-v1.5（Bi-Encoder，33M 参数）
│                                        # BAAI/bge-reranker-base（Cross-Encoder，279M 参数）
│
├── thesis_outline.md                       # 论文大纲（完整五章目录 + 配图计划）
└── README.md                               # 本文件
```

---

## 系统架构

```
用户查询
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 1: 混合检索                    │
│  BM25（稀疏关键词）+ FAISS（稠密语义）│
│  ＋ 倒数秩融合 RRF (k=60)            │
│  → Top-20 候选文档池                  │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Phase 2: 语义重排序                  │
│  Cross-Encoder (bge-reranker-base)   │
│  深层注意力交互打分 → Sigmoid 置信度  │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Phase 3: 动态截断                    │
│  置信度阈值 θ=0.3，最多保留 5 条      │
│  自适应过滤噪声文档                   │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Phase 4: LLM 生成                    │
│  通义千问 qwen-plus                   │
│  temperature=0.1，max_tokens=512     │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Phase 5: RAGAS 评估                  │
│  Faithfulness / Answer Relevancy     │
│  Context Precision / Context Recall   │
└──────────────────────────────────────┘
```

---

## 环境配置

### 第一步：克隆项目

```bash
git clone <your-repo-url>
cd 毕业论文实验
```

### 第二步：安装依赖

```bash
# 方式一：一键安装（推荐）
pip install -r requirements.txt

# 方式二：分步安装（使用清华镜像加速）
pip install pandas numpy matplotlib jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install rank_bm25 sentence-transformers faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ragas datasets langchain-huggingface -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第三步：配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 DashScope API Key
# DASHSCOPE_API_KEY=your-actual-api-key
```

> **API Key 申请**：前往 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 注册并获取 API Key。

### 第四步：下载模型（自动缓存）

运行任意 notebook 或脚本时，模型将自动从 HuggingFace 镜像下载至 `hf_models/` 目录。如遇下载问题，可手动设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com       # Windows PowerShell
```

---

## 快速开始

### 执行顺序

| 顺序 | 文件 | 说明 |
|:---:|---|---|
| 1 | `notebooks/01_data_exploration.ipynb` | 下载原始数据并构建平板知识库 |
| 2 | `notebooks/02_build_indices_and_baseline.ipynb` | 构建 BM25 和 FAISS 双路索引 |
| 3 | `notebooks/03_advanced_rag_pipeline.ipynb` | 交互式体验抗噪 RAG 核心管道 |
| 4 | `notebooks/04_llm_generation.ipynb` | 端到端生成测试 |
| 5 | `notebooks/05_ragas_evaluation.ipynb` | Baseline vs Robust 双通道批量评测 |
| 6 | `notebooks/06_ablation_study.ipynb` | 消融实验 |
| 7 | `notebooks/07_noise_robustness.ipynb` | 噪声鲁棒性实验 |
| 8 | `notebooks/08_param_sensitivity.ipynb` | 参数敏感性分析 |

> 每个阶段的输出是下一个阶段的输入，请务必按顺序执行。

### 使用 Python 脚本（可选）

```python
from scripts.rag_api import build_baseline, build_robust, generate_answer

# Baseline 一键检索
ctxs = build_baseline("平板屏幕清晰度怎么样")
answer = generate_answer("平板屏幕清晰度怎么样", ctxs, model='baseline')

# Noise-Robust 一键检索
ctxs = build_robust("平板屏幕清晰度怎么样")
answer = generate_answer("平板屏幕清晰度怎么样", ctxs, model='robust')
```

---

## 核心实验结果

### 主实验对比（RAGAS 四指标，200 条 QA）

| 模型 | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|:---:|:---:|:---:|:---:|
| **Noise-Robust RAG** | **0.864** | **0.792** | **0.831** | **0.806** |
| Baseline RAG | 0.712 | 0.684 | 0.618 | 0.741 |

### 消融实验

| 模型（变体） | Faithfulness | Answer Relevancy | Context Precision |
|---|:---:|:---:|:---:|
| 完整模型 | 0.864 | 0.792 | 0.831 |
| 去混合检索组 | 0.821 | 0.751 | 0.764 |
| 去重排序组 | 0.768 | 0.723 | 0.681 |
| 去动态截断组 | 0.826 | 0.773 | 0.736 |

---

## 论文附录索引

以下文档可作为论文附录的原始素材：

| 附录 | 内容 | 位置 |
|---|---|---|
| 附录 A | 核心算法实现代码 | `scripts/03_advanced_rag_pipeline.py` |
| 附录 B | 环境配置与依赖清单 | `requirements.txt` + `docs/rag_api_design.md` |
| 附录 C | 数据预处理详细流程 | `notebooks/01_data_exploration.ipynb` |
| 附录 D | Prompt 工程文档 | `docs/prompts/README.md` |
| 附录 E | RAGAS 评估 Prompt 模板 | `docs/prompts/README.md` §3 |
| 附录 F | QA 数据集构建详细说明 | `docs/prompts/README.md` §1 |
| 附录 G | RAG API 设计文档 | `docs/rag_api_design.md` |
| 附录 H | 降噪实验完整结果数据 | `results/ablation_results.csv` |
| 附录 I | 参数敏感性实验完整数值 | `results/param_sensitivity_results.csv` |
| 附录 J | 噪声鲁棒性实验完整数据 | `results/noise_robustness_results.csv` |
| 附录 K | 典型案例详细分析 | `results/case_study_results.csv` |
| 附录 L | 完整 QA 测试集（200 条） | `data/qa_dataset.csv` |

---

## 技术栈

| 组件 | 选型 | 用途 |
|---|---|---|
| 数据处理 | Pandas, NumPy | 数据加载、清洗、统计分析 |
| 中文分词 | jieba | BM25 索引分词 |
| 稀疏检索 | rank_bm25 (BM25Okapi) | 关键词匹配召回 |
| 稠密检索 | FAISS (IndexFlatIP) | 向量语义检索 |
| 向量化模型 | `BAAI/bge-small-zh-v1.5` | 512 维文本向量编码 |
| 重排模型 | `BAAI/bge-reranker-base` | 细粒度语义打分 |
| 大语言模型 | 通义千问 `qwen-plus` | 答案生成（OpenAI 兼容接口）|
| 评估框架 | RAGAS | 四指标自动化评估 |
| 可视化 | Matplotlib | 论文配图 |

---

## 致谢

- 数据集：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
- 向量模型 & 重排模型：[BAAI (北京智源人工智能研究院)](https://huggingface.co/BAAI)
- LLM API：[阿里云 DashScope](https://dashscope.aliyuncs.com/)
- 评估框架：[RAGAS](https://github.com/explodinggradients/ragas)
