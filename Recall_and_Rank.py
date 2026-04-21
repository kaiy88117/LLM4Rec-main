from tools.recommend_tool import format_recommendations, recommend_videos


def main() -> None:
    query = "science fiction action movies"
    results = recommend_videos(query, k=5)
    print("Query:", query)
    print(format_recommendations(results))


if __name__ == "__main__":
    main()
