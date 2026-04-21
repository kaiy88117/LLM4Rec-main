# run_chat.py

from agents.recommender_agent import get_recommender_agent

def main():
    agent = get_recommender_agent()
    print("🎬 欢迎使用视频推荐助手！请输入你的兴趣，比如：我想看猫猫搞笑视频\n")

    while True:
        user_input = input("🧑 你：")
        if user_input.lower() in ["exit", "quit", "退出", "q"]:
            print("👋 感谢使用，再见！")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"🤖 推荐助手：\n{response}\n")
        except Exception as e:
            print("❌ 出错了：", e)

if __name__ == "__main__":
    main()
