import os
from typing import List

from dotenv import load_dotenv

from project_config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    HumanMessage = None


def _build_llm():
    if ChatOpenAI is None or HumanMessage is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    kwargs = {
        "temperature": 0.2,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "openai_api_key": api_key,
    }
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        kwargs["openai_api_base"] = api_base
    return ChatOpenAI(**kwargs)


LLM = _build_llm()


def generate_explanation(user_tags: List[str], video_tags: List[str], score: float, title: str) -> str:
    matched = sorted(set(user_tags) & set(video_tags))
    if matched:
        joined = " / ".join(matched)
        return f"这部电影和你的偏好重合度最高的类型是 {joined}，因此被排在前面。"

    if not video_tags:
        return f"《{title}》在当前候选结果里综合得分较高，因此被推荐。"

    if LLM is None:
        return f"《{title}》属于 {' / '.join(video_tags[:3])} 等类型，在当前候选结果里综合得分较高。"

    prompt = (
        f"用户偏好类型: {', '.join(user_tags) or '未明确'}\n"
        f"推荐电影: {title}\n"
        f"电影类型: {', '.join(video_tags)}\n"
        f"综合分数: {score:.4f}\n"
        "请生成一句中文推荐理由，要求自然、具体、不要夸张。"
    )
    try:
        response = LLM.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        return f"《{title}》属于 {' / '.join(video_tags[:3])} 等类型，在当前候选结果里综合得分较高。"
