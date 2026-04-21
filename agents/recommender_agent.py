import os

from dotenv import load_dotenv

from project_config import LLM4RecError, PROJECT_ROOT
from tools.recommend_tool import format_recommendations, recommend_videos

load_dotenv(PROJECT_ROOT / ".env")

try:
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain.tools import tool
    from langchain_openai import ChatOpenAI
except ImportError:
    AgentType = None
    Tool = None
    initialize_agent = None
    tool = None
    ChatOpenAI = None


class DirectRecommenderAgent:
    def invoke(self, payload: dict[str, str]) -> str:
        query = payload.get("input", "")
        results = recommend_videos(query)
        return format_recommendations(results)


def _build_llm_agent():
    if not all([ChatOpenAI, Tool, initialize_agent, AgentType, tool]):
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    @tool
    def recommend(user_input: str) -> str:
        """Recommend movies from a natural-language preference query."""
        try:
            return format_recommendations(recommend_videos(user_input))
        except LLM4RecError as exc:
            return str(exc)

    kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "openai_api_key": api_key,
    }
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        kwargs["openai_api_base"] = api_base

    llm = ChatOpenAI(**kwargs)
    tools = [
        Tool.from_function(
            func=recommend,
            name="MovieRecommender",
            description="Recommend MovieLens movies from a user preference query.",
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
    )


def get_recommender_agent():
    return _build_llm_agent() or DirectRecommenderAgent()
