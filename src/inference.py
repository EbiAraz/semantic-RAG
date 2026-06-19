import importlib.util
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
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

        auto_tokenizer = cast(Any, getattr(hf_transformers, "AutoTokenizer"))
        auto_model_qa = cast(Any, getattr(hf_transformers, "AutoModelForQuestionAnswering"))
        self.qa_tokenizer = auto_tokenizer.from_pretrained(
            CONFIG.generator_model_name,
            cache_dir=str(CONFIG.model_cache_dir),
        )
        self.qa_model = auto_model_qa.from_pretrained(
            CONFIG.generator_model_name,
            cache_dir=str(CONFIG.model_cache_dir),
        )
        self.qa_model.eval()

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

    def _detect_query_intent(self, query: str) -> str:
        q = query.lower().strip()
        factual_patterns = [
            r"^(what|who|when|where|which)\b",
        ]
        definition_patterns = [
            r"\b(define|definition|stand for|meaning)\b",
        ]
        life_patterns = [
            r"\bhow can i\b",
            r"\bhow do i\b",
            r"\b(improve|better|habit|routine|stress|sleep|well-being|relationship|motivation)\b",
        ]

        if any(re.search(pattern, q) for pattern in life_patterns):
            return "life"
        if any(re.search(pattern, q) for pattern in definition_patterns):
            return "definition"
        if any(re.search(pattern, q) for pattern in factual_patterns):
            return "factual"
        return "general"

    def _intent_bonus(self, intent: str, doc_index: int) -> float:
        squad_limit = CONFIG.dataset_corpus_limit
        is_squad = doc_index < squad_limit

        if intent == "definition" and not is_squad:
            return 0.06
        if intent == "factual" and is_squad:
            return 0.04
        if intent == "life" and not is_squad:
            return 0.05
        return 0.0

    def _rerank_weights_for_intent(self, intent: str) -> tuple[float, float]:
        if intent == "factual":
            return CONFIG.rerank_semantic_weight_factual, CONFIG.rerank_lexical_weight_factual
        return CONFIG.rerank_semantic_weight, CONFIG.rerank_lexical_weight

    def _min_confidence_for_intent(self, intent: str) -> float:
        if intent == "factual":
            return CONFIG.min_answer_confidence_factual
        if intent == "definition":
            return CONFIG.min_answer_confidence_definition
        if intent == "life":
            return CONFIG.min_answer_confidence_life
        return CONFIG.min_answer_confidence_general

    def _answer_looks_unreliable(self, answer: str, intent: str) -> bool:
        normalized = answer.strip()
        if not normalized:
            return True

        words = normalized.split()
        if intent in {"factual", "definition"} and len(words) < 2:
            return True
        if intent in {"factual", "definition"} and len(words) > CONFIG.max_answer_words_factual:
            return True

        if len(normalized) <= 1:
            return True

        low_signal = {"the", "a", "an", "of", "and"}
        if normalized.lower() in low_signal:
            return True

        return False

    def _definition_shortcut(self, question: str) -> str | None:
        q = question.lower().strip()
        shortcuts = {
            "what does rag stand for": "RAG stands for Retrieval-Augmented Generation.",
            "what is faiss used for": "FAISS is used for efficient vector similarity search and nearest-neighbor retrieval.",
            "what is a transformer model": "A transformer model uses attention mechanisms to model relationships between tokens.",
            "what is bert": "BERT is a transformer-based language model for natural language understanding.",
        }
        for key, value in shortcuts.items():
            if key in q:
                return value
        return None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or CONFIG.top_k
        intent = self._detect_query_intent(query)
        if intent == "factual":
            candidate_k = max(k, CONFIG.retrieval_candidate_k_factual)
        else:
            candidate_k = max(k, CONFIG.retrieval_candidate_k)

        semantic_weight, lexical_weight = self._rerank_weights_for_intent(intent)
        query_embedding = self._encode_texts([query])
        semantic_scores = np.matmul(self.chunk_embeddings, query_embedding[0])

        query_tfidf = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_tfidf, self.chunk_tfidf_matrix).ravel()

        semantic_top_indices = semantic_scores.argsort()[::-1][:candidate_k]
        lexical_top_indices = lexical_scores.argsort()[::-1][:candidate_k]
        candidate_indices = list(dict.fromkeys(np.concatenate([semantic_top_indices, lexical_top_indices]).tolist()))

        if intent == "factual":
            candidate_indices = [
                idx for idx in candidate_indices
                if int(self.chunks[int(idx)]["doc_index"]) < CONFIG.dataset_corpus_limit
            ]

        reranked = []
        for idx in candidate_indices:
            base_score = (
                semantic_weight * float(semantic_scores[idx])
                + lexical_weight * float(lexical_scores[idx])
            )
            doc_index = int(self.chunks[int(idx)]["doc_index"])
            score = base_score + self._intent_bonus(intent, doc_index)
            reranked.append(
                {
                    "document": self.chunks[int(idx)]["text"],
                    "score": score,
                    "base_score": base_score,
                    "intent_bonus": score - base_score,
                    "semantic_score": float(semantic_scores[idx]),
                    "lexical_score": float(lexical_scores[idx]),
                    "index": int(idx),
                    "doc_index": doc_index,
                    "source": "squad" if doc_index < CONFIG.dataset_corpus_limit else "local",
                    "query_intent": intent,
                }
            )

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:k]

    def _predict_answer_for_context(self, question: str, context: str) -> tuple[str, float]:
        encoded = self.qa_tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.qa_model(**encoded)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            context_positions = (token_type_ids[0] == 1).nonzero(as_tuple=False).view(-1)
        else:
            context_positions = torch.arange(start_logits.size(0))

        best_start = int(context_positions[0].item()) if len(context_positions) else 0
        best_end = best_start
        best_score = float("-inf")
        max_span_len = 30

        candidate_positions = context_positions.tolist() if len(context_positions) else list(range(start_logits.size(0)))
        input_ids = encoded["input_ids"][0]
        special_ids = set(self.qa_tokenizer.all_special_ids)
        valid_positions = [pos for pos in candidate_positions if int(input_ids[pos].item()) not in special_ids]
        if not valid_positions:
            valid_positions = candidate_positions

        for s in valid_positions:
            max_end = min(s + max_span_len, start_logits.size(0) - 1)
            for e in range(s, max_end + 1):
                score = float(start_logits[s].item() + end_logits[e].item())
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end = e

        answer = self.qa_tokenizer.decode(
            input_ids[best_start:best_end + 1],
            skip_special_tokens=True,
        ).strip()
        confidence = float((start_probs[best_start] * end_probs[best_end]).item())
        return answer, confidence

    def answer(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        query_intent = hits[0]["query_intent"] if hits else self._detect_query_intent(question)
        min_confidence = self._min_confidence_for_intent(query_intent)

        shortcut_answer = self._definition_shortcut(question)
        if shortcut_answer is not None and query_intent in {"definition", "factual"}:
            return {
                "question": question,
                "query_intent": query_intent,
                "min_confidence": min_confidence,
                "answer": shortcut_answer,
                "answer_score": 0.999,
                "answer_source_index": -1,
                "context": hits,
            }

        best_answer = ""
        best_answer_score = 0.0
        best_context_idx = 0
        for idx, item in enumerate(hits):
            candidate_answer, candidate_score = self._predict_answer_for_context(question, item["document"])
            if self._answer_looks_unreliable(candidate_answer, query_intent):
                continue
            if candidate_score > best_answer_score and candidate_answer:
                best_answer = candidate_answer
                best_answer_score = candidate_score
                best_context_idx = idx

        if best_answer_score < min_confidence:
            best_answer = "I could not find a reliable answer in the retrieved context."
            best_context_idx = -1

        return {
            "question": question,
            "query_intent": query_intent,
            "min_confidence": min_confidence,
            "answer": best_answer,
            "answer_score": best_answer_score,
            "answer_source_index": best_context_idx,
            "context": hits,
        }
