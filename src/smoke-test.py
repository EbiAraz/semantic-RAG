import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import RAGEngine


def main() -> None:
    engine = RAGEngine()
    questions = [
        "What does RAG stand for?",
        "What is the capital of France?",
        "Who invented relativity?",
        "What city is the capital of Iran?",
        "How can I reduce stress?",
        "not good",
    ]

    print("Running Semantic RAG smoke test...\n")
    failures = 0
    for idx, question in enumerate(questions, start=1):
        result = engine.answer(question, top_k=3)
        answer = str(result.get("answer", "")).strip()
        mode = str(result.get("answer_mode", "unknown"))
        score = float(result.get("answer_score", 0.0))

        print(f"{idx}. Q: {question}")
        print(f"   A: {answer}")
        print(f"   mode={mode}, score={score:.4f}\n")

        if not answer:
            failures += 1

    if failures > 0:
        raise SystemExit(f"Smoke test failed: {failures} empty answers")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
