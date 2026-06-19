import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers as hf_transformers

from config import CONFIG


def _load_data_loader_module():
    module_path = Path(__file__).with_name("data-loader.py")
    spec = importlib.util.spec_from_file_location("data_loader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load data-loader.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RAGEngine:
    def __init__(self) -> None:
        CONFIG.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_loader = _load_data_loader_module()
        self.embedding_model = SentenceTransformer(
            CONFIG.embedding_model_name,
            cache_folder=str(CONFIG.model_cache_dir),
        )

        qa_pipeline = cast(Any, getattr(hf_transformers, "pipeline"))
        self.reader = qa_pipeline(
            "question-answering",
            model=CONFIG.generator_model_name,
            tokenizer=CONFIG.generator_model_name,
            model_kwargs={"cache_dir": str(CONFIG.model_cache_dir)},
        )

        docs = self._data_loader.load_corpus()
        if len(docs) < CONFIG.min_documents:
            raise RuntimeError("Corpus is too small to build a useful RAG index.")

        self.documents = docs
        self.chunks = self._load_or_create_chunks(self.documents)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_tfidf_matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])
        self.chunk_embeddings = self._load_or_create_embeddings(self.chunks)

    def _chunk_documents(self, documents: list[str]) -> list[dict]:
        chunks = []
        chunk_size = CONFIG.chunk_size_words
        overlap = CONFIG.chunk_overlap_words

        for doc_index, document in enumerate(documents):
            words = document.split()
            if not words:
                continue

            if len(words) <= chunk_size:
                chunks.append({"text": document, "doc_index": doc_index})
                continue

            step = max(1, chunk_size - overlap)
            for start in range(0, len(words), step):
                chunk_words = words[start:start + chunk_size]
                if not chunk_words:
                    continue
                chunks.append({
                    "text": " ".join(chunk_words),
                    "doc_index": doc_index,
                })
                if start + chunk_size >= len(words):
                    break

        return chunks

    def _load_or_create_chunks(self, documents: list[str]) -> list[dict]:
        chunks_path = CONFIG.processed_chunks_path
        if chunks_path.exists():
            return json.loads(chunks_path.read_text(encoding="utf-8"))

        chunks = self._chunk_documents(documents)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        return chunks

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _load_or_create_embeddings(self, chunks: list[dict]) -> np.ndarray:
        embeddings_path = CONFIG.processed_chunk_embeddings_path
        if embeddings_path.exists():
            cached_embeddings = np.load(embeddings_path)
            if cached_embeddings.shape[0] == len(chunks):
                return cached_embeddings

        embeddings = self._encode_texts([chunk["text"] for chunk in chunks])
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        return embeddings

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or CONFIG.top_k
        candidate_k = max(k, CONFIG.retrieval_candidate_k)
        query_embedding = self._encode_texts([query])
        semantic_scores = np.matmul(self.chunk_embeddings, query_embedding[0])

        query_tfidf = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_tfidf, self.chunk_tfidf_matrix).ravel()

        semantic_top_indices = semantic_scores.argsort()[::-1][:candidate_k]
        reranked = []
        for idx in semantic_top_indices:
            combined_score = (
                CONFIG.rerank_semantic_weight * float(semantic_scores[idx])
                + CONFIG.rerank_lexical_weight * float(lexical_scores[idx])
            )
            reranked.append(
                {
                    "document": self.chunks[int(idx)]["text"],
                    "score": combined_score,
                    "semantic_score": float(semantic_scores[idx]),
                    "lexical_score": float(lexical_scores[idx]),
                    "index": int(idx),
                    "doc_index": int(self.chunks[int(idx)]["doc_index"]),
                }
            )

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:k]

    def answer(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        context = "\n".join([item["document"] for item in hits])
        qa = self.reader(question=question, context=context)

        return {
            "question": question,
            "answer": qa.get("answer", "").strip(),
            "answer_score": float(qa.get("score", 0.0)),
            "context": hits,
        }
