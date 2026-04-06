"""
rag_api.py — 统一的 RAG 实验 API 入口
=========================================
提供所有 notebook 和脚本共同依赖的 API 封装，
确保实验流程一致性和代码复用。

导入方式：
    from rag_api import NoiseRobustRAG, BaselineRAG, RAGGenerator, load_rag_config

核心 API：
    load_rag_config()       → 返回各路径配置字典
    build_baseline(query)   → Baseline RAG 检索
    build_robust(query)     → Noise-Robust RAG 检索
    generate_answer(query, contexts, model='baseline') → LLM 生成答案
"""

import os
import sys

# 配置 HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./hf_models"

import pandas as pd
import numpy as np
import jieba
import pickle
import faiss
import torch
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

# 抑制 jieba 初始化日志
logging.getLogger('jieba').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
def load_rag_config():
    """
    返回项目路径配置字典。
    所有路径均为相对于项目根目录的路径。
    """
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        'base_dir': base,
        'corpus_path': os.path.join(base, 'data/processed/tablet_corpus.csv'),
        'bm25_path':    os.path.join(base, 'indices/bm25_model.pkl'),
        'faiss_path':   os.path.join(base, 'indices/faiss_index.bin'),
        'qa_path':      os.path.join(base, 'qa_dataset.csv'),
        'results_dir':  os.path.join(base, 'results'),
    }


# ---------------------------------------------------------------------------
# RAG 模型类（从 03_advanced_rag_pipeline.py 封装）
# ---------------------------------------------------------------------------
class NoiseRobustRAG:
    """
    Noise-Robust RAG（完整模型）
    三阶段管线：混合检索 → Cross-Encoder 重排序 → 动态截断
    """
    def __init__(self, config=None):
        config = config or load_rag_config()
        print("[NoiseRobustRAG] Initializing...")
        self.df_corpus = pd.read_csv(config['corpus_path'])
        self.corpus_texts = self.df_corpus['review'].tolist()

        with open(config['bm25_path'], "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(config['faiss_path'])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[NoiseRobustRAG] Device: {self.device}")
        self.bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=self.device)
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=self.device)
        print("[NoiseRobustRAG] Ready.\n")

    def hybrid_search(self, query, top_k=20, k_rrf=60):
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True)
        _, faiss_top = self.faiss_index.search(q_emb, top_k)
        faiss_top = faiss_top[0].tolist()

        rrf_scores = {}
        for rank, doc_id in enumerate(bm25_top):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
        for rank, doc_id in enumerate(faiss_top):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [x[0] for x in sorted_rrf]

    def semantic_rerank(self, query, candidate_indices):
        candidate_docs = [self.corpus_texts[idx] for idx in candidate_indices]
        pairs = [[query, doc] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)
        probs = 1 / (1 + np.exp(-scores))
        return sorted(zip(candidate_indices, probs), key=lambda x: x[1], reverse=True)

    def dynamic_truncate(self, sorted_candidates, threshold=0.3, max_docs=5):
        final_docs = []
        for doc_id, score in sorted_candidates:
            if score < threshold or len(final_docs) >= max_docs:
                break
            final_docs.append((self.corpus_texts[doc_id], float(score)))
        if not final_docs and sorted_candidates:
            doc_id, score = sorted_candidates[0]
            final_docs.append((self.corpus_texts[doc_id], float(score)))
        return final_docs

    def retrieve(self, query, top_k=20, k_rrf=60, threshold=0.3, max_docs=5):
        """
        完整检索管线。
        返回: list of (text, score) tuples
        """
        candidates = self.hybrid_search(query, top_k=top_k, k_rrf=k_rrf)
        reranked = self.semantic_rerank(query, candidates)
        return self.dynamic_truncate(reranked, threshold=threshold, max_docs=max_docs)


class BaselineRAG:
    """
    Baseline RAG（朴素 RAG）
    仅使用 FAISS 向量检索，取固定 Top-K
    """
    def __init__(self, config=None):
        config = config or load_rag_config()
        print("[BaselineRAG] Initializing...")
        self.df_corpus = pd.read_csv(config['corpus_path'])
        self.corpus_texts = self.df_corpus['review'].tolist()

        self.faiss_index = faiss.read_index(config['faiss_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=self.device)
        print(f"[BaselineRAG] Device: {self.device}, Ready.\n")

    def retrieve(self, query, top_k=5):
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True)
        _, indices = self.faiss_index.search(q_emb, top_k)
        indices = indices[0].tolist()
        return [(self.corpus_texts[idx], 1.0) for idx in indices]


# ---------------------------------------------------------------------------
# LLM 生成（从 04_llm_generation.py 封装）
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGGenerator:
    """
    RAG 答案生成器。
    支持真实 OpenAI API 调用或本地模拟。
    """
    def __init__(self, model_name='qwen-plus'):
        self.model_name = model_name
        self.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        self.use_simulated = not self.api_key or not OPENAI_AVAILABLE

        if self.use_simulated:
            print(f"[RAGGenerator] Using SIMULATED mode (no API key).")
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            print(f"[RAGGenerator] Using REAL API: {model_name}")

    def generate(self, query, contexts, reference='', model='baseline'):
        """
        生成答案。

        参数:
            query:     用户查询
            contexts:  检索到的上下文 list[(text, score)]
            reference: 参考答案（可选，用于评测）
            model:     'baseline' 或 'robust'
        返回:
            answer:    str
        """
        if self.use_simulated:
            return self._simulate_answer(query, contexts, model)

        context_texts = [ctx[0] for ctx in contexts]
        context_block = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_texts)])

        system_prompt = (
            "你是一个严格基于给定证据进行回答的助手。"
            "如果证据中的信息不足以回答问题，你应当明确指出这一点，"
            "而不是编造信息。回答必须简洁、专业、有条理。"
        )

        user_prompt = f"""根据以下证据回答用户问题。

证据：
{context_block}

用户问题：{query}

回答要求：
1. 仅使用上述证据中的信息进行回答
2. 如果证据不足，请明确说明
3. 避免提及"根据证据"等表述，直接给出答案
4. 禁止编造证据中不存在的信息
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RAGGenerator] API call failed: {e}, falling back to simulation.")
            return self._simulate_answer(query, contexts, model)

    def _simulate_answer(self, query, contexts, model):
        """模拟生成答案（用于调试或无API环境）"""
        n = len(contexts)
        if model == 'baseline':
            return (
                f"[Baseline] 根据检索到的 {n} 条信息，"
                f"关于「{query[:15]}...」的回答如下："
                f"{contexts[0][0][:30]}...（详见上下文）"
            )
        else:
            return (
                f"[Noise-Robust] 经去噪处理后，基于 {n} 条高置信度证据，"
                f"关于「{query[:15]}...」给出以下专业回答："
                f"{contexts[0][0][:30]}...（详见上下文）"
            )


# ---------------------------------------------------------------------------
# 消融模型工厂
# ---------------------------------------------------------------------------
class AblationVariant:
    """
    消融实验变体工厂。
    通过组合不同模块开关生成对应的检索器。
    """
    def __init__(self, config=None, use_hybrid=True, use_rerank=True,
                 use_truncate=True, threshold=0.3, max_docs=5):
        self.config = config or load_rag_config()
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.use_truncate = use_truncate
        self.threshold = threshold
        self.max_docs = max_docs

        # 初始化共享组件
        self.df_corpus = pd.read_csv(self.config['corpus_path'])
        self.corpus_texts = self.df_corpus['review'].tolist()

        with open(self.config['bm25_path'], "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(self.config['faiss_path'])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=self.device)
        if use_rerank:
            self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=self.device)

    def retrieve(self, query, top_k=20, k_rrf=60):
        if self.use_hybrid:
            # 混合检索
            tokenized = list(jieba.cut(query))
            bm25_s = self.bm25.get_scores(tokenized)
            bm25_top = np.argsort(bm25_s)[::-1][:top_k].tolist()
            q_emb = self.bi_encoder.encode([query], normalize_embeddings=True)
            _, faiss_top = self.faiss_index.search(q_emb, top_k)
            faiss_top = faiss_top[0].tolist()

            rrf_scores = {}
            for rank, did in enumerate(bm25_top):
                rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (k_rrf + rank + 1)
            for rank, did in enumerate(faiss_top):
                rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (k_rrf + rank + 1)
            candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            candidate_indices = [x[0] for x in candidates]
        else:
            # 仅 FAISS
            q_emb = self.bi_encoder.encode([query], normalize_embeddings=True)
            _, faiss_top = self.faiss_index.search(q_emb, top_k)
            candidate_indices = faiss_top[0].tolist()

        if self.use_rerank:
            docs = [self.corpus_texts[idx] for idx in candidate_indices]
            pairs = [[query, doc] for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            probs = 1 / (1 + np.exp(-scores))
            sorted_candidates = sorted(zip(candidate_indices, probs),
                                       key=lambda x: x[1], reverse=True)
        else:
            # 直接返回，不重排（取前 max_docs）
            sorted_candidates = [(idx, 1.0) for idx in candidate_indices]

        if self.use_truncate:
            final = []
            for did, score in sorted_candidates:
                if score < self.threshold or len(final) >= self.max_docs:
                    break
                final.append((self.corpus_texts[did], float(score)))
            if not final and sorted_candidates:
                did, score = sorted_candidates[0]
                final.append((self.corpus_texts[did], float(score)))
            return final
        else:
            # 固定 Top-5
            return [(self.corpus_texts[did], float(score))
                    for did, score in sorted_candidates[:self.max_docs]]


# ---------------------------------------------------------------------------
# 快捷调用函数
# ---------------------------------------------------------------------------
_baseline_rag = None
_robust_rag = None
_generator = None


def get_baseline():
    global _baseline_rag
    if _baseline_rag is None:
        _baseline_rag = BaselineRAG()
    return _baseline_rag


def get_robust():
    global _robust_rag
    if _robust_rag is None:
        _robust_rag = NoiseRobustRAG()
    return _robust_rag


def get_generator():
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


def build_baseline(query):
    """Baseline RAG 一键检索"""
    return get_baseline().retrieve(query)


def build_robust(query, top_k=20, k_rrf=60, threshold=0.3, max_docs=5):
    """Noise-Robust RAG 一键检索"""
    return get_robust().retrieve(query, top_k=top_k, k_rrf=k_rrf,
                                  threshold=threshold, max_docs=max_docs)


def generate_answer(query, contexts, model='baseline'):
    """LLM 生成答案"""
    return get_generator().generate(query, contexts, model=model)


if __name__ == '__main__':
    cfg = load_rag_config()
    print("RAG API Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # 测试 Baseline
    print("\n[Test] Baseline RAG retrieval:")
    ctxs = build_baseline("平板屏幕清晰度怎么样")
    print(f"  Retrieved {len(ctxs)} contexts")
    for i, (text, score) in enumerate(ctxs):
        print(f"  [{i+1}] score={score:.3f} | {text[:30]}...")

    # 测试 Noise-Robust
    print("\n[Test] Noise-Robust RAG retrieval:")
    ctxs = build_robust("平板屏幕清晰度怎么样")
    print(f"  Retrieved {len(ctxs)} contexts")
    for i, (text, score) in enumerate(ctxs):
        print(f"  [{i+1}] score={score:.3f} | {text[:30]}...")
