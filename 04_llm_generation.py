import os
from openai import OpenAI

class RAGGenerator:
    """
    面向事实一致性的 RAG 生成模块
    负责将检索到的高质量上下文与 User Query 融合成 Prompt，并调用 LLM 生成最终答案。
    """
    def __init__(self, api_key, base_url, model_name):
        # 增加明确的 timeout 和 max_retries 防止网络请求无响应导致进度条永久卡死
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=3)
        self.model_name = model_name

    def build_prompt(self, query, context_list):
        """
        构建防幻觉的 System Prompt
        """
        # 兼容 context_list 是 list of strings 或 list of tuples 的情况
        if len(context_list) > 0 and isinstance(context_list[0], tuple):
            context_str = "\n".join([f"证据 {i+1}: {text}" for i, (text, score) in enumerate(context_list)])
        else:
            context_str = "\n".join([f"证据 {i+1}: {text}" for i, text in enumerate(context_list)])
            
        system_prompt = (
            "你是一个严谨的电商智能客服助手。\n"
            "【核心任务】：请严格基于以下提供的【真实用户评论证据】来回答用户的问题。\n"
            "【严格约束】：\n"
            "1. 你的回答必须100%忠实于提供的证据，绝不能捏造、发散或使用你的内在先验知识。\n"
            "2. 如果提供的证据中没有包含回答问题所需的信息，请直接回答“根据已有信息，无法得出结论”，绝不能强行猜测。\n"
            "3. 综合证据中的多方观点，给出客观、中立的总结。"
        )
        
        user_prompt = f"【真实用户评论证据】:\n{context_str}\n\n【用户提问】: {query}\n\n请给出你的回答："
        return system_prompt, user_prompt

    def generate_answer(self, query, context_list):
        """
        调用 LLM 生成答案
        """
        if not context_list:
            return "抱歉，知识库中没有检索到与您问题相关的高质量事实依据。"
            
        system_prompt, user_prompt = self.build_prompt(query, context_list)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM 调用失败: {e}"
