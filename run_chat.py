from agents.recommender_agent import get_recommender_agent
from project_config import LLM4RecError


def main():
    agent = get_recommender_agent()
    print("MovieLens recommender is ready.")
    print("Example: sci-fi action movies / comedy movies / Tom Hanks movies")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"Assistant:\n{response}\n")
        except LLM4RecError as exc:
            print(f"Error: {exc}\n")
        except Exception as exc:
            print(f"Unexpected error: {exc}\n")


if __name__ == "__main__":
    main()
