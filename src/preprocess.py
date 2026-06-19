import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from config import CONFIG


def _load_data_loader_module():
	module_path = Path(__file__).with_name("data-loader.py")
	spec = importlib.util.spec_from_file_location("data_loader", module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError("Could not load data-loader.py")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _write_json_list(path: Path, items: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _encode_chunks(chunks: list[dict]) -> np.ndarray:
	model = SentenceTransformer(
		CONFIG.embedding_model_name,
		cache_folder=str(CONFIG.model_cache_dir),
	)
	embeddings = model.encode(
		[chunk["text"] for chunk in chunks],
		normalize_embeddings=True,
		convert_to_numpy=True,
	)
	return np.asarray(embeddings, dtype=np.float32)


def _chunk_documents(documents: list[str]) -> list[dict]:
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
			chunks.append({"text": " ".join(chunk_words), "doc_index": doc_index})
			if start + chunk_size >= len(words):
				break

	return chunks


def preprocess(limit: int | None = None) -> dict:
	data_loader = _load_data_loader_module()
	corpus_limit = limit or CONFIG.dataset_corpus_limit
	question_limit = limit or CONFIG.dataset_question_limit

	corpus = data_loader.load_corpus_from_dataset(limit=corpus_limit)
	questions = data_loader.load_questions_from_dataset(limit=question_limit)
	chunks = _chunk_documents(corpus)
	embeddings = _encode_chunks(chunks)

	_write_json_list(CONFIG.processed_corpus_path, corpus)
	_write_json_list(CONFIG.processed_questions_path, questions)
	CONFIG.processed_chunks_path.parent.mkdir(parents=True, exist_ok=True)
	CONFIG.processed_chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
	np.save(CONFIG.processed_chunk_embeddings_path, embeddings)

	return {
		"dataset": CONFIG.dataset_name,
		"corpus_count": len(corpus),
		"question_count": len(questions),
		"chunk_count": len(chunks),
		"embedding_count": int(embeddings.shape[0]),
		"corpus_path": str(CONFIG.processed_corpus_path),
		"questions_path": str(CONFIG.processed_questions_path),
		"chunks_path": str(CONFIG.processed_chunks_path),
		"embeddings_path": str(CONFIG.processed_chunk_embeddings_path),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Download and preprocess the SQuAD dataset into local cache files.")
	parser.add_argument("--limit", type=int, default=None, help="Optional limit for both corpus and questions.")
	args = parser.parse_args()

	report = preprocess(limit=args.limit)
	print(f"Saved {report['corpus_count']} corpus passages to {report['corpus_path']}")
	print(f"Saved {report['question_count']} questions to {report['questions_path']}")
	print(f"Saved {report['chunk_count']} chunks to {report['chunks_path']}")
	print(f"Saved {report['embedding_count']} chunk embeddings to {report['embeddings_path']}")


if __name__ == "__main__":
	main()
