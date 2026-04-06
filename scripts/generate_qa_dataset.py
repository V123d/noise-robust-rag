import pandas as pd
import json
from openai import OpenAI
import time
import os

# 配置与 04/05 相同的大模型 API 信息
API_KEY = "sk-8c998af7bd73446683c969fcee175a6c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus" 

def generate_qa_pairs():
    print("正在加载语料库...")
    df = pd.read_csv("./data/processed/tablet_corpus.csv")
    reviews = df['review'].tolist()
    
    # 抽取具有代表性的长文本给大模型，避免极端冗长导致失忆
    # 3000条评论通常约 5 万个 Token，完全在 qwen-plus 的舒适处理区间
    if len(reviews) > 3000:
        reviews = pd.Series(reviews).sample(n=3000, random_state=42).tolist()
    
    context_text = "\n".join([f"评论{i+1}: {rev}" for i, rev in enumerate(reviews)])
    # 截断硬保护
    context_text = context_text[:80000] 

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_qa_pairs = []
    
    print(f"🚀 开始调用大模型 API ({MODEL_NAME}) 批量生成 QA 对...")
    # 循环 4 次，每次索要 50 条，总共产生约 200 条
    for batch in range(4):
        print(f"🔄 正在生成第 {batch+1}/4 批 (目标进度 {len(all_qa_pairs)+50}/200)...")
        
        prompt = (
            "你现在是一位极度苛刻、严谨的科研数据标注专家。我为你提供了一批包含丰富细节的真实平板买家评论。\n"
            f"你的任务是：深度挖掘这批评论里的细节事实，生成 50 条用于双盲评估的中文问答对，这是你的第 {batch+1} 批次任务。\n"
            "【生成约束条件】：\n"
            f"1. 请务必挖掘不同维度的用户疑问（例如屏幕显示、性能游戏、电池续航、发热、品控、物流售后等），不要重复。\n"
            "2. user_input：模拟电商用户极其真实、口语化的提问。\n"
            "3. reference：作为 Ground Truth（金标准），它必须是100%基于下方评论内容提炼出的客观陈述。如果评论中包含不同的正负反馈，需综合陈述，绝不可自己编造。\n"
            "4. 强制返回格式：不要包含任何多余的开场白或解释代码块，请直接返回一个严谨的 JSON 数组结构。\n"
            "【希望输出的 JSON 结构参考如下】：\n"
            "[\n"
            "  {\"user_input\": \"这台平板屏幕长时间看刺眼吗？\", \"reference\": \"根据用户反馈，屏幕显示清晰并具备护眼模式，暂无大量认为极其刺眼的不良反馈。\"}\n"
            "]\n\n"
            f"【供你参考的极长买家评论片段库】：\n{context_text}"
        )
        
        try:
            # 使用略微提升的 temperature 保证每批次产出的多样性，不至于问答雷同
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严格遵循指令的 JSON 数据序列化机器，只能输出由对象组成的 JSON Array。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6 + (batch * 0.1), 
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理可能附带的 Markdown 代码块残余
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            batch_qas = json.loads(content.strip())
            all_qa_pairs.extend(batch_qas)
            print(f"✅ 第 {batch+1} 批成功获取 {len(batch_qas)} 条问答数据。")
            
        except json.JSONDecodeError:
            print(f"❌ 第 {batch+1} 批 JSON 解析失败，大模型回复格式错误: {content[:100]}...")
        except Exception as e:
            print(f"❌ 第 {batch+1} 批 API 调用出错: {e}")
            
        # 防止触发 API 的速率限制 (Rate Limit)
        time.sleep(3)
        
    # 保存结果
    if all_qa_pairs:
        out_df = pd.DataFrame(all_qa_pairs)
        # 如果大模型偶尔多生成或少生成几条，这里去重截断到前200条
        out_df = out_df.drop_duplicates(subset=['user_input']).head(200) 
        output_path = "qa_dataset.csv"
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\\n🎉 大功告成！已成功将 {len(out_df)} 条高质量测试题保存至根目录的 '{output_path}'")
        print("现在您可以直接一键运行 05_ragas_evaluation.ipynb 了。")
    else:
        print("⚠️ 未能生成任何数据，请检查您的 API 余额或网络连接情况。")

if __name__ == '__main__':
    generate_qa_pairs()
