import json
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

from config import CONFIG


def _clean_texts(texts: list[str]) -> list[str]:
    return [text.strip() for text in texts if text and text.strip()]


def _unique_preserve_order(texts: list[str]) -> list[str]:
    seen = set()
    unique_texts = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique_texts.append(text)
    return unique_texts


def load_local_documents(path: Path) -> list[str]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return _clean_texts([str(item) for item in data])
        if isinstance(data, dict):
            return _clean_texts([str(value) for value in data.values()])
        return []

    return _clean_texts(path.read_text(encoding="utf-8").splitlines())


def load_corpus_from_dataset(
    dataset_name: str | None = None,
    split: str | None = None,
    context_field: str | None = None,
    limit: int | None = None,
) -> list[str]:
    dataset_name = dataset_name or CONFIG.dataset_name
    split = split or CONFIG.dataset_corpus_split
    context_field = context_field or CONFIG.dataset_context_field
    limit = limit or CONFIG.dataset_corpus_limit

    dataset = cast(Any, load_dataset(dataset_name, split=split))
    contexts = _clean_texts([str(value) for value in dataset[context_field]])
    contexts = _unique_preserve_order(contexts)
    return contexts[:limit]


def load_questions_from_dataset(
    dataset_name: str | None = None,
    split: str | None = None,
    question_field: str | None = None,
    limit: int | None = None,
) -> list[str]:
    dataset_name = dataset_name or CONFIG.dataset_name
    split = split or CONFIG.dataset_question_split
    question_field = question_field or CONFIG.dataset_question_field
    limit = limit or CONFIG.dataset_question_limit

    dataset = cast(Any, load_dataset(dataset_name, split=split))
    questions = _clean_texts([str(value) for value in dataset[question_field]])
    questions = _unique_preserve_order(questions)
    return questions[:limit]


def load_corpus() -> list[str]:
    processed_corpus = load_local_documents(CONFIG.processed_corpus_path)
    if processed_corpus:
        return processed_corpus

    try:
        documents = load_corpus_from_dataset()
        if documents:
            return documents
    except Exception:
        pass

    local_documents = load_local_documents(CONFIG.data_path)
    if local_documents:
        return local_documents

    return [
        "Istanbul is the largest city in Turkey.",
        "Artificial intelligence is a branch of computer science.",
        "Python is a popular programming language for machine learning.",
        "Retrieval-Augmented Generation combines search with text generation.",
        "FAISS is a library for efficient similarity search.",
    ]


def load_questions() -> list[str]:
    processed_questions = load_local_documents(CONFIG.processed_questions_path)
    if processed_questions:
        return processed_questions

    try:
        questions = load_questions_from_dataset()
        if questions:
            return questions
    except Exception:
        pass

    fallback_questions_path = CONFIG.data_path.with_name("questions.txt")
    questions = load_local_documents(fallback_questions_path)
    if questions:
        return questions

    return [
        "What does RAG stand for?",
        "What is FAISS used for?",
        "What city is the capital of Iran?",
    ]
