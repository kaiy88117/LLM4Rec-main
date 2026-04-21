# agents/recommender_agent.py

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/carterhe/Desktop/LLM4Rec/.env")

print("🧪 DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("🧪 DEBUG: OPENAI_API_BASE =", os.getenv("OPENAI_API_BASE"))

from tools.recommend_tool import recommend_videos
# from langchain_community.chat_models import ChatOpenAI
# ✅ 正确的写法（支持 openai_api_base）
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool


@tool
def recommend(user_input: str) -> list[dict]:
    """推荐视频：接受用户自然语言输入，如“我想看猫猫搞笑视频”，内部解析为关键词。"""
    tags = user_input.replace("，", ",").replace("、", ",").replace(" ", ",").split(",")
    tags = [t.strip() for t in tags if t.strip()]
    print("🧪 提取关键词:", tags)
    return recommend_videos(tags)


tools = [
    Tool.from_function(
        func=recommend,
        name="VideoRecommender",
        description=(
            "推荐视频工具，输入参数是一个关键词列表，例如 ['猫猫', '搞笑']。\n"
            "用户会输入自然语言描述兴趣，你应该从中提取关键词，格式化为一个字符串列表。\n"
            "⚠️ 注意：不要传入字符串，而是 list[str]。\n"
            "示例：用户说“我想看猫猫搞笑视频” → 输入应为 ['猫猫', '搞笑']"
        )
    )
]



print("🧪 LLM initialized with:")
print("   - API_KEY =", os.getenv("OPENAI_API_KEY"))
print("   - BASE_URL =", os.getenv("OPENAI_API_BASE"))

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)



agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

def get_recommender_agent():
    return agent


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    print("🔍 Testing LLM directly...")

    try:
        response = llm.invoke([HumanMessage(content="你是谁？")])
        print("✅ 模型返回：", response.content)
    except Exception as e:
        print("❌ 模型调用失败：", e)
