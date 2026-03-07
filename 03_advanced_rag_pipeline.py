import os
# --- 极其重要：配置 Hugging Face 国内镜像加速下载 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# --- 新增：将模型下载缓存路径更改为当前项目文件夹下 ---
os.environ["HF_HOME"] = "./hf_models" 

import pandas as pd
import numpy as np
import jieba
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import time

class NoiseRobustRAG:
    """
    面向电商领域事实一致性的抗噪 RAG 核心模型
    对应论文第三章的核心架构设计
    """
    def __init__(self, corpus_path, bm25_path, faiss_path):
        print("正在初始化抗噪 RAG 管道...")
        
        # 1. 加载知识库
        self.df_corpus = pd.read_csv(corpus_path)
        self.corpus_texts = self.df_corpus['review'].tolist()
        
        # 2. 加载底层索引 (Phase 2 的产物)
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(faiss_path)
        
        # 3. 加载向量化模型 (Bi-Encoder) 和 重排模型 (Cross-Encoder)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用计算设备: {self.device}")
        
        self.bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=self.device)
        # 使用 bge-reranker-base 作为交叉编码器，权衡了速度和精度
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=self.device)
        print("✅ 模型与索引加载完毕！\n")

    def hybrid_search(self, query, top_k=20):
        """
        模块一：混合检索 (Hybrid Search) 与 倒数秩融合 (RRF)
        作用：扩大召回池，防止单一检索方式漏掉关键信息。
        """
        # -- BM25 召回 --
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # -- FAISS 召回 --
        query_embedding = self.bi_encoder.encode([query], normalize_embeddings=True)
        _, faiss_top_indices = self.faiss_index.search(query_embedding, top_k)
        faiss_top_indices = faiss_top_indices[0]
        
        # -- 倒数秩融合 (Reciprocal Rank Fusion, RRF) --
        # 公式: RRF_score = 1 / (k + rank), 这里的常数 k 通常设为 60
        rrf_scores = {}
        k_rrf = 60 
        
        for rank, doc_id in enumerate(bm25_top_indices):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
            
        for rank, doc_id in enumerate(faiss_top_indices):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
            
        # 根据 RRF 得分重新排序，取最终的 top_k 进入重排序池
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidate_indices = [x[0] for x in sorted_rrf]
        
        return candidate_indices

    def semantic_rerank(self, query, candidate_indices):
        """
        模块二：基于 Cross-Encoder 的语义重排序
        作用：让 Query 和 Doc 进行深层注意力交互，挤出反事实和无关噪声。
        """
        candidate_docs = [self.corpus_texts[idx] for idx in candidate_indices]
        
        # 构造 Cross-Encoder 所需的输入对: [[query, doc1], [query, doc2], ...]
        sentence_pairs = [[query, doc] for doc in candidate_docs]
        
        # 计算打分 (BGE Reranker 输出的是 logits)
        rerank_scores = self.cross_encoder.predict(sentence_pairs)
        
        # 将 logits 转换为 0-1 之间的概率值 (Sigmoid)
        probabilities = 1 / (1 + np.exp(-rerank_scores))
        
        # 降序排列
        sorted_combined = sorted(zip(candidate_indices, probabilities), key=lambda x: x[1], reverse=True)
        return sorted_combined

    def dynamic_truncate(self, sorted_candidates, threshold=0.3, max_docs=5):
        """
        模块三：动态截断机制 (Dynamic Truncation)
        核心创新：不固定取 Top-K，而是根据分数断层和置信度硬截断噪声。
        """
        final_docs = []
        for doc_id, score in sorted_candidates:
            # 停止条件：如果分数低于最低置信阈值，或者达到了最大上下文限制
            if score < threshold or len(final_docs) >= max_docs:
                break
            final_docs.append((self.corpus_texts[doc_id], score))
            
        # 如果所有文档分数都很低，为了防止无话可说，至少保留得分最高的1个
        if len(final_docs) == 0 and len(sorted_candidates) > 0:
             doc_id, score = sorted_candidates[0]
             final_docs.append((self.corpus_texts[doc_id], score))
             
        return final_docs

    def retrieve_context(self, query):
        """
        整合完整的抗噪检索 Pipeline
        """
        # 1. 混合检索召回候选集 (Top-20)
        candidates = self.hybrid_search(query, top_k=20)
        
        # 2. Cross-Encoder 细粒度重排
        reranked_candidates = self.semantic_rerank(query, candidates)
        
        # 3. 动态截断去除尾部噪声
        final_context = self.dynamic_truncate(reranked_candidates)
        
        return final_context

class BaselineRAG:
    """
    基准 RAG 模型 (Naive RAG)
    仅使用向量检索 (Dense Retrieval) 取 Top-K
    """
    def __init__(self, corpus_path, faiss_path):
        print("正在初始化基准 RAG 管道 (FAISS)...")
        # 1. 加载知识库
        self.df_corpus = pd.read_csv(corpus_path)
        self.corpus_texts = self.df_corpus['review'].tolist()
        
        # 2. 加载向量索引
        self.faiss_index = faiss.read_index(faiss_path)
        
        # 3. 加载向量化模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=self.device)
        print("✅ 基准模型与索引加载完毕！\n")

    def retrieve_context(self, query, top_k=5):
        """
        仅使用向量检索获取 Context
        """
        query_embedding = self.bi_encoder.encode([query], normalize_embeddings=True)
        _, faiss_top_indices = self.faiss_index.search(query_embedding, top_k)
        faiss_top_indices = faiss_top_indices[0]
        
        # 返回与 NoiseRobustRAG 相同的结构：(text, score) 或只是 text
        # 为了兼容 RAGGenerator，这里返回包含文本的列表，score 假定为 1.0 (或只传text，RAGGenerator已兼容)
        final_context = [(self.corpus_texts[idx], 1.0) for idx in faiss_top_indices]
        return final_context

