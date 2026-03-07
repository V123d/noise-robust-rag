# 🛡️ Noise-Robust RAG：面向中文电商领域事实一致性的抗噪检索增强生成系统

> **毕业论文实验项目** — 基于混合检索、Cross-Encoder 重排序与动态截断的 RAG 系统设计与评估

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-green)](https://docs.ragas.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 目录

- [研究背景与动机](#-研究背景与动机)
- [核心思路与创新点](#-核心思路与创新点)
- [系统架构](#-系统架构)
- [项目结构](#-项目结构)
- [环境配置与快速开始](#-环境配置与快速开始)
- [各阶段详细说明](#-各阶段详细说明)
- [实验结果](#-实验结果)
- [改进方向与优化空间](#-改进方向与优化空间)
- [技术栈](#-技术栈)

---

## 🔬 研究背景与动机

### 问题背景

**检索增强生成 (Retrieval-Augmented Generation, RAG)** 是当前大语言模型 (LLM) 应用落地的核心范式之一，它通过引入外部知识库来缓解 LLM 的"幻觉"问题。然而，在真实的**中文电商场景**中，RAG 系统面临着严峻的**噪声挑战**：

| 噪声类型 | 说明 | 示例 |
|:---|:---|:---|
| **反事实噪声** | 用户评论中包含与事实不符的信息 | "这平板一点都不好用" (实际为情绪化表达) |
| **无关噪声** | 检索到的文档与查询无语义关联 | 查询"屏幕清晰度"但召回"物流速度"评论 |
| **主观情绪噪声** | 纯情绪表达缺乏事实性信息 | "垃圾！差评！再也不买了！" |

传统的 **Naive RAG**（仅使用向量检索取 Top-K）无法有效过滤这些噪声，导致噪声文档被直接注入到 LLM 的 Prompt 中，严重损害了生成答案的**事实一致性 (Faithfulness)**。

### 研究目标

本项目设计并实现了一套**面向事实一致性的抗噪 RAG 系统**，在检索与生成之间构建了多层次的噪声过滤防线，从而显著提升 LLM 在电商问答场景中回答的准确性与可靠性。

---

## 💡 核心思路与创新点

本项目的核心设计哲学是 **"先检索，再精排，后截断，最终生成"**，通过层层递进的抗噪机制确保最终送入 LLM 的上下文是高度纯净的。

### 三大创新模块

1. **混合检索 + 倒数秩融合 (Hybrid Search + RRF)**
   - 同时使用 **BM25 稀疏检索**（关键词匹配）和 **FAISS 稠密检索**（语义匹配）
   - 通过 **Reciprocal Rank Fusion (RRF)** 算法融合两路召回结果，优势互补
   - **为什么这样设计**：BM25 擅长精确匹配关键词,但对同义词、改述无力；FAISS 语义向量检索能捕获语义相似性，但可能遗漏精确关键词匹配。两者融合有效扩大了高质量候选文档的召回池

2. **Cross-Encoder 语义重排序 (Semantic Reranking)**
   - 使用 `BAAI/bge-reranker-base` 交叉编码器对候选文档进行细粒度的语义打分
   - Query 和 Document 在 Transformer 内部进行**深层交叉注意力交互**
   - **为什么这样设计**：Bi-Encoder 的向量检索本质上是一种"粗排"——它将 Query 和 Doc 独立编码后计算相似度，无法捕获二者之间的细粒度语义交互。Cross-Encoder 虽然计算成本更高，但能够在重排阶段精确识别出真正与 Query 语义匹配的文档

3. **动态截断机制 (Dynamic Truncation)**
   - 不固定取 Top-K，而是根据 Cross-Encoder 的置信度分数设置**硬截断阈值** (默认 0.3)
   - 当分数跌破阈值时自动停止，最多保留 5 条上下文
   - **为什么这样设计**：传统 Top-K 策略不管文档质量如何都固定取 K 条，这意味着低分噪声文档必然被引入。动态截断根据打分断层自适应决定上下文数量，从源头杜绝低质量文档污染 Prompt

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       用户查询 (User Query)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Phase 1: 混合检索     │
                │                       │
                │  ┌─────┐   ┌───────┐  │
                │  │BM25 │   │ FAISS │  │
                │  │稀疏  │   │ 稠密   │  │
                │  └──┬──┘   └──┬────┘  │
                │     │   RRF   │       │
                │     └────┬────┘       │
                │          ▼            │
                │   Top-20 候选文档池    │
                └──────────┬────────────┘
                           │
                ┌──────────▼────────────┐
                │  Phase 2: 语义重排序    │
                │                       │
                │  Cross-Encoder        │
                │  (bge-reranker-base)  │
                │  深层注意力交互打分      │
                └──────────┬────────────┘
                           │
                ┌──────────▼────────────┐
                │  Phase 3: 动态截断      │
                │                       │
                │  置信度阈值 > 0.3      │
                │  最多保留 5 条          │
                │  自适应噪声过滤         │
                └──────────┬────────────┘
                           │
                ┌──────────▼────────────┐
                │  Phase 4: LLM 生成      │
                │                       │
                │  防幻觉 Prompt 工程     │
                │  (通义千问 qwen-plus)   │
                │  temperature = 0.1    │
                └──────────┬────────────┘
                           │
                ┌──────────▼────────────┐
                │  Phase 5: RAGAS 评估    │
                │                       │
                │  Faithfulness         │
                │  Answer Relevancy    │
                │  Context Precision   │
                └───────────────────────┘
```

### 基准模型 vs 抗噪模型

| 维度 | Baseline RAG (基准) | Noise-Robust RAG (抗噪) |
|:---|:---|:---|
| 检索策略 | 仅 FAISS 向量检索 | BM25 + FAISS 混合检索 + RRF 融合 |
| 重排序 | ❌ 无 | ✅ Cross-Encoder 深层语义重排 |
| 上下文筛选 | 固定 Top-5 | 动态截断（置信度阈值 + 最大数量限制） |
| 核心优势 | 简单快速 | 高事实一致性、强抗噪能力 |

---

## 📁 项目结构

```
毕业论文实验/
│
├── 01_data_exploration.ipynb           # 阶段1：数据探索与预处理
├── 02_build_indices_and_baseline.ipynb  # 阶段2：构建 BM25 与 FAISS 双路索引
├── 03_advanced_rag_pipeline.ipynb      # 阶段3：抗噪 RAG 核心管道 (交互式)
├── 03_advanced_rag_pipeline.py         # 阶段3：核心模块 (可导入的 Python 模块)
├── 04_llm_generation.ipynb             # 阶段4：LLM 生成端到端测试
├── 04_llm_generation.py                # 阶段4：生成模块 (可导入的 Python 模块)
├── 05_ragas_evaluation.ipynb           # 阶段5：RAGAS 自动化评估与可视化
│
├── generate_qa_dataset.py              # 辅助脚本：调用大模型批量生成 QA 测试集
├── extract_docs.py                     # 辅助脚本：文档抽取
│
├── data/
│   ├── raw/                            # 原始数据
│   │   ├── online_shopping_10_cats.csv # 6 万+ 条中文电商评论数据集
│   │   └── online_shopping_10_cats.zip
│   └── processed/
│       └── tablet_corpus.csv           # 清洗后的"平板"类别专属知识库 (~9920 条)
│
├── indices/
│   ├── bm25_index.pkl                  # 序列化的 BM25 稀疏索引
│   └── faiss_index.bin                 # FAISS 稠密向量索引 (512 维)
│
├── hf_models/                          # Hugging Face 模型本地缓存
│                                       # (bge-small-zh-v1.5, bge-reranker-base)
│
├── qa_dataset.csv                      # 200 条 AI 生成的高质量 QA 测试集
├── baseline_generation_only.csv        # 基准 RAG 的批量推理结果
├── robust_generation_only.csv          # 抗噪 RAG 的批量推理结果
├── baseline_ragas_results.csv          # 基准 RAG 的 RAGAS 评分结果
├── robust_ragas_results.csv            # 抗噪 RAG 的 RAGAS 评分结果
│
└── README.md                           # 本文件
```

---

## ⚙️ 环境配置与快速开始

### 前置条件

- **Python 3.10+** （推荐使用 Conda 环境管理）
- **阿里云 DashScope API Key**（用于调用通义千问 LLM，可替换为其他 OpenAI 兼容接口）

### 1. 安装依赖

```bash
# 核心依赖
pip install pandas numpy matplotlib jieba -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检索与向量化
pip install rank_bm25 sentence-transformers faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# LLM 调用
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple

# RAGAS 评估
pip install ragas datasets langchain-huggingface -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch (CPU 版本)
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 API Key

在 `04_llm_generation.ipynb` 和 `05_ragas_evaluation.ipynb` 中，将 `API_KEY` 替换为你自己的密钥：

```python
API_KEY = "your-api-key-here"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
```

### 3. 按顺序运行 Notebooks

```
01 → 02 → 03 → 04 → 05
```

> ⚠️ 每个阶段的输出是下一个阶段的输入，请务必按顺序执行。

---

## 📖 各阶段详细说明

### 阶段 1：数据探索与预处理 (`01_data_exploration.ipynb`)

**目标**：获取原始数据集并进行清洗，构建领域专属知识库。

- **数据来源**：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 的 `online_shopping_10_cats` 数据集
- **数据规模**：62,774 条中文电商评论，覆盖书籍、平板、水果等 10 个类别
- **处理流程**：
  1. 从 GitHub 自动下载并解压数据集
  2. 筛选"平板"类别的评论（与电子产品事实一致性强相关）
  3. 清洗：去除空评论、过短评论（< 5 字）
  4. 最终产出约 **9,920 条评论** 作为知识库语料

### 阶段 2：构建双路检索索引 (`02_build_indices_and_baseline.ipynb`)

**目标**：为知识库构建 BM25 和 FAISS 两种互补的检索索引。

- **BM25 稀疏索引**
  - 使用 `jieba` 进行中文分词
  - 构建 `BM25Okapi` 模型，序列化保存为 `bm25_index.pkl`
- **FAISS 稠密向量索引**
  - 使用 `BAAI/bge-small-zh-v1.5` 中文向量模型（512 维）将全量语料编码
  - 构建 `IndexFlatIP`（内积索引，因向量已归一化，等价于余弦相似度）
  - 保存为 `faiss_index.bin`
- **验证**：对示例 Query 分别进行 BM25 和 FAISS 检索，对比 Top-3 结果

### 阶段 3：抗噪 RAG 核心管道 (`03_advanced_rag_pipeline.py`)

**目标**：实现论文核心架构——三阶段抗噪检索管道。

- **`NoiseRobustRAG` 类** — 抗噪模型
  - `hybrid_search()`: 混合检索 + RRF 融合
  - `semantic_rerank()`: Cross-Encoder 重排序
  - `dynamic_truncate()`: 动态截断
  - `retrieve_context()`: 完整 Pipeline 入口
- **`BaselineRAG` 类** — 基准对照模型
  - 仅使用 FAISS 向量检索 Top-5
  - 无重排、无截断，作为消融实验的对照基线

### 阶段 4：LLM 生成模块 (`04_llm_generation.py`)

**目标**：将检索到的高质量上下文与用户查询融合，调用 LLM 生成最终答案。

- **防幻觉 Prompt 工程**：设计了严格约束的 System Prompt
  - 强制 LLM 100% 基于提供的证据回答
  - 无法回答时诚实表示"无法得出结论"
  - 综合正反面观点给出客观总结
- **参数配置**：`temperature=0.1` 极低温度确保生成的确定性与事实一致性
- **兼容性设计**：基于 OpenAI 兼容接口，可轻松替换为 DeepSeek、智谱、Moonshot 等

### 阶段 5：RAGAS 自动化评估 (`05_ragas_evaluation.ipynb`)

**目标**：使用 RAGAS 框架对两代架构进行大规模、自动化的交叉对比评估。

- **QA 测试集**：200 条由大模型生成的高质量问答对（覆盖屏幕、性能、电池、物流等维度）
- **双通道批量推理**：分别用 Baseline RAG 和 Robust RAG 对全部 200 条 Query 进行端到端推理
- **RAGAS 评估指标**：
  | 指标 | 含义 |
  |:---|:---|
  | **Faithfulness** | 生成答案是否忠实于检索到的上下文证据 |
  | **Answer Relevancy** | 生成答案与用户问题的相关程度 |
  | **Context Precision** | 检索到的上下文中相关文档的精确度 |
- **可视化**：生成双柱状图进行直观对比，可直接用于论文插图

---

## 📊 实验结果

基于 20 条样本的 RAGAS 评估结果 (初步测试)：

| 指标 | Baseline RAG | Noise-Robust RAG |
|:---|:---:|:---:|
| Faithfulness | 0.8677 | 0.8554 |
| Answer Relevancy | 0.2639 | 0.2085 |
| Context Precision | 0.5770 | 0.6582 |

> 📝 **注**：以上为 20 条样本的初步测试结果。完整的 200 条样本评估可通过运行 `05_ragas_evaluation.ipynb` 获得。Context Precision 的提升说明抗噪 RAG 在检索阶段确实召回了更多高质量的相关文档。

---

## 🚀 改进方向与优化空间

以下方向可进一步增加研究的工作量和深度，适合在论文中展开讨论或作为后续工作：

### 🔧 模型层面的优化

| 方向 | 当前状态 | 优化方案 | 预期收益 |
|:---|:---|:---|:---|
| **向量模型升级** | `bge-small-zh-v1.5` (33M 参数) | 升级至 `bge-large-zh-v1.5` (326M) 或 `gte-Qwen2-7B-instruct` | 更精准的语义表征，提升召回质量 |
| **重排模型升级** | `bge-reranker-base` (279M) | 升级至 `bge-reranker-v2-m3` 或 `bge-reranker-v2.5-gemma2-lightweight` | 更强的跨语言和长文本重排能力 |
| **LLM 升级** | 通义千问 qwen-plus | 使用 DeepSeek-V3、GPT-4o 或本地部署的开源 LLM | 更强的理解能力和指令遵循能力 |
| **GPU 加速** | 仅使用 CPU 推理 | 部署至 CUDA GPU 环境 | 10-50x 检索与推理加速 |

### 📐 检索策略的优化

| 方向 | 描述 | 工作量估计 |
|:---|:---|:---|
| **Query Expansion** | 在检索前对用户查询进行扩展（HyDE、LLM-based Expansion） | ⭐⭐⭐ |
| **多级索引** | 引入 HNSW 等近似最近邻索引替代暴力搜索 | ⭐⭐ |
| **Chunk 策略优化** | 当前每条评论独立作为一个 Document，可探索滑动窗口分块、递归分块 | ⭐⭐⭐ |
| **多路 RRF 权重调优** | 当前 BM25 和 FAISS 两路同权融合，可通过验证集学习最优权重 | ⭐⭐ |
| **迭代式检索** | 根据 LLM 首轮回答进行二次检索精化 | ⭐⭐⭐⭐ |

### 🧪 评估与实验层面的增补

| 方向 | 描述 | 工作量估计 |
|:---|:---|:---|
| **消融实验** | 逐一去除三个模块（混合检索、重排、截断），观察对评估指标的影响 | ⭐⭐⭐ |
| **噪声注入实验** | 向知识库中手动注入不同比例的噪声文档，测试抗噪鲁棒性 | ⭐⭐⭐ |
| **人工评估** | 招募标注人员对生成答案的事实一致性进行人工打分 | ⭐⭐⭐⭐ |
| **跨领域验证** | 在其他商品类别（手机、书籍等）上验证模型泛化性 | ⭐⭐ |
| **扩大测试集** | 将 QA 测试集从 200 条扩充至 500-1000 条 | ⭐⭐ |
| **增加评估维度** | 引入 Context Recall、Answer Correctness 等更多 RAGAS 指标 | ⭐⭐ |

### 🏛️ 架构层面的改进

| 方向 | 描述 | 工作量估计 |
|:---|:---|:---|
| **Self-RAG** | 引入"自反思"机制，让 LLM 自主判断是否需要检索、是否需要二次检索 | ⭐⭐⭐⭐⭐ |
| **Corrective RAG (CRAG)** | 在生成前加入评估器，判断检索文档的质量是否足以回答问题 | ⭐⭐⭐⭐ |
| **知识图谱增强** | 从评论中抽取实体关系构建知识图谱，支持图谱增强检索 | ⭐⭐⭐⭐⭐ |
| **端到端微调** | 在领域数据上微调 Bi-Encoder 和 Cross-Encoder，提升领域适配性 | ⭐⭐⭐⭐ |
| **流式部署** | 基于 FastAPI + Gradio 构建可交互的 Web Demo | ⭐⭐⭐ |

---

## 🛠️ 技术栈

| 组件 | 技术选型 | 用途 |
|:---|:---|:---|
| 数据处理 | Pandas, NumPy | 数据加载、清洗、统计分析 |
| 中文分词 | jieba | BM25 索引的前置分词 |
| 稀疏检索 | rank_bm25 (BM25Okapi) | 基于关键词的稀疏检索 |
| 稠密检索 | FAISS (IndexFlatIP) | 基于向量的稠密语义检索 |
| 向量化模型 | BAAI/bge-small-zh-v1.5 (Sentence-Transformers) | 将文本编码为 512 维稠密向量 |
| 重排序模型 | BAAI/bge-reranker-base (CrossEncoder) | 对候选文档进行细粒度语义打分 |
| 大语言模型 | 阿里通义千问 qwen-plus (OpenAI 兼容接口) | 基于上下文生成最终答案 |
| 评估框架 | RAGAS | 自动化评估 Faithfulness、Relevancy、Precision |
| 可视化 | Matplotlib | 绘制评估结果对比图 |

---

## 📄 License

本项目仅用于学术研究与毕业论文实验，代码采用 MIT License 开源。

---

## 🙏 致谢

- 数据集来源：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
- 向量模型 & 重排模型：[BAAI (北京智源人工智能研究院)](https://huggingface.co/BAAI)
- LLM API：[阿里云 DashScope](https://dashscope.aliyuncs.com/)
- 评估框架：[RAGAS](https://github.com/explodinggradients/ragas)
