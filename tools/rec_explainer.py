from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

def generate_explanation(user_tags: List[str], video_tags: List[str], score: float, title: str) -> str:
    # 尝试模板生成
    matched = list(set(user_tags) & set(video_tags))
    if matched:
        return f"你喜欢的关键词 {matched} 与该视频“{title}”的主题高度相关，因此推荐给你。"

    # 否则使用 GPT fallback
    prompt = (
        f"用户兴趣标签：{', '.join(user_tags)}\n"
        f"推荐视频标题：{title}\n"
        f"视频标签：{', '.join(video_tags)}\n"
        f"请为推荐结果生成一句自然语言理由："
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print("⚠️ GPT 生成失败：", e)
        return f"该视频“{title}”与你的兴趣相似，因此推荐给你。"
