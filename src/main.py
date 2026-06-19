from inference import RAGEngine


def main() -> None:
    engine = RAGEngine()
    print("RAG is ready. Type your question, or type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = engine.answer(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['answer_score']:.4f}")
        print("Retrieved context:")
        for item in result["context"]:
            print(f"- score={item['score']:.4f} | {item['document']}")


if __name__ == "__main__":
    main()
