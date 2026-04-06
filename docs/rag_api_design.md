# RAG API 设计文档

本文档描述 `scripts/rag_api.py` 的设计架构、核心类接口与使用示例，可作为论文附录 J（完整代码清单）的辅助说明。

---

## 1. 整体设计目标

`rag_api.py` 是项目的**统一 API 入口**，旨在为所有 notebook 和脚本提供一致的接口封装，确保：

1. **检索配置统一**：索引路径、模型名称等硬编码值集中在一处
2. **模块可复用**：`NoiseRobustRAG`、`BaselineRAG`、`RAGGenerator` 可独立使用或组合使用
3. **模拟模式兼容**：无 API Key 时自动降级为模拟模式，便于离线调试
4. **消融实验支持**：通过 `AblationVariant` 工厂类灵活生成不同模块组合的变体

---

## 2. 核心配置

### `load_rag_config() → dict`

返回项目路径配置字典，所有路径相对于项目根目录：

```python
{
    'base_dir':     'D:/毕业论文实验',          # 项目根目录
    'corpus_path':  'data/processed/tablet_corpus.csv',
    'bm25_path':    'indices/bm25_model.pkl',
    'faiss_path':   'indices/faiss_index.bin',
    'qa_path':      'qa_dataset.csv',
    'results_dir':  'results',
}
```

> **注意**：`bm25_index.pkl` 在 notebook 2 中构建时文件名可能为 `bm25_model.pkl`，两者等价。

---

## 3. 核心类

### 3.1 `NoiseRobustRAG`（完整抗噪 RAG）

继承自 `scripts/03_advanced_rag_pipeline.py`，是实验组的核心检索器。

#### `__init__(config=None)`

加载语料库、BM25 索引、FAISS 索引，并初始化 Bi-Encoder 和 Cross-Encoder 模型。

| 组件 | 模型 | 设备 |
|---|---|---|
| Bi-Encoder | `BAAI/bge-small-zh-v1.5` | CUDA/CPU 自动检测 |
| Cross-Encoder | `BAAI/bge-reranker-base` | CUDA/CPU 自动检测 |

#### `hybrid_search(query, top_k=20, k_rrf=60) → list[int]`

**输入**：用户查询，候选数量，RRF 融合常数

**处理流程**：
1. BM25 稀疏检索 → 取 top_k 条
2. FAISS 向量检索 → 取 top_k 条
3. RRF 融合并排序 → 返回 doc_id 列表

**RRF 公式**：

$$
\text{RRF}_{score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}
$$

#### `semantic_rerank(query, candidate_indices) → list[tuple[int, float]]`

**输入**：查询与候选 doc_id 列表

**处理流程**：
1. 将候选文档与查询拼接为 `[query, doc]` 对
2. Cross-Encoder 预测 logit 分数
3. Sigmoid 归一化到 [0, 1] 置信度
4. 按置信度降序排列

**Sigmoid 转换**：

$$
p = \frac{1}{1 + e^{-\text{logits}}}
$$

#### `dynamic_truncate(sorted_candidates, threshold=0.3, max_docs=5) → list[tuple[str, float]]`

**输入**：已排序的候选列表

**处理流程**：
1. 遍历候选列表，遇到分数 < threshold 或已保留 max_docs 条时停止
2. 若所有分数均低于阈值，至少保留得分最高的一条（兜底机制）
3. 返回 `(文档文本, 置信度分数)` 元组列表

#### `retrieve(query, top_k=20, k_rrf=60, threshold=0.3, max_docs=5) → list[tuple[str, float]]`

完整三阶段管线，返回最终上下文列表。

---

### 3.2 `BaselineRAG`（基准朴素 RAG）

对照组——仅使用 FAISS 向量检索，取固定 Top-5，无重排序、无截断。

| 模块 | BaselineRAG | NoiseRobustRAG |
|---|---|---|
| 检索方式 | 仅 FAISS | BM25 + FAISS 双路 |
| 候选数量 | 固定 5 | 默认 20（RRF融合后） |
| 重排序 | ❌ | ✅ Cross-Encoder |
| 动态截断 | ❌ | ✅ 置信度阈值 0.3 |
| 上下文条数 | 固定 5 | 自适应 1~5 |

---

### 3.3 `RAGGenerator`（LLM 答案生成器）

#### `__init__(model_name='qwen-plus')`

- 若环境变量 `DASHSCOPE_API_KEY` 未设置，自动切换到模拟模式
- 模拟模式下 `generate()` 返回结构化占位文本，不调用任何 API

#### `generate(query, contexts, reference='', model='baseline') → str`

**参数**：
- `query`：用户原始查询
- `contexts`：`list[tuple[str, float]]`，检索结果
- `reference`：Ground Truth 参考答案（用于评测）
- `model`：`'baseline'` 或 `'robust'`（仅影响模拟模式输出格式）

**流程**：
1. 构造 Context Block（编号证据列表）
2. 组装 System Prompt + User Prompt
3. 调用 DashScope API（`temperature=0.3`, `max_tokens=512`）
4. 失败时自动降级为模拟模式

---

### 3.4 `AblationVariant`（消融实验工厂）

通过开关参数生成不同的消融模型变体：

```python
# 仅移除混合检索（退化为 Baseline 逻辑）
AblationVariant(use_hybrid=False, use_rerank=True, use_truncate=True)

# 仅移除重排序
AblationVariant(use_hybrid=True, use_rerank=False, use_truncate=True)

# 仅移除动态截断
AblationVariant(use_hybrid=True, use_rerank=True, use_truncate=False)

# 完整模型（等价于 NoiseRobustRAG）
AblationVariant(use_hybrid=True, use_rerank=True, use_truncate=True)
```

所有变体共享相同的组件初始化逻辑，内部通过布尔开关控制各阶段的执行。

---

## 4. 快捷函数

```python
from rag_api import build_baseline, build_robust, generate_answer

# Baseline 一键检索
ctxs = build_baseline("平板屏幕清晰度怎么样")

# Noise-Robust 一键检索（可自定义参数）
ctxs = build_robust("平板屏幕清晰度怎么样", top_k=20, k_rrf=60, threshold=0.3, max_docs=5)

# LLM 生成
answer = generate_answer("平板屏幕清晰度怎么样", ctxs, model='robust')
```

---

## 5. 环境变量依赖

| 变量名 | 用途 | 来源 |
|---|---|---|
| `DASHSCOPE_API_KEY` | LLM API 调用 | `.env` 文件或系统环境变量 |
| `HF_ENDPOINT` | HuggingFace 镜像 | 国内推荐设为 `https://hf-mirror.com` |
| `HF_HOME` | 模型本地缓存路径 | 默认为 `./hf_models` |

---

## 更新记录

| 版本 | 日期 | 更新内容 |
|---|---|---|
| 1.0 | 2026-04-06 | 初始版本 |
