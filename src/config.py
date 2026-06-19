from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
	project_root: Path = Path(__file__).resolve().parents[1]
	data_path: Path = project_root / "data" / "documents.txt"
	processed_corpus_path: Path = project_root / "data" / "squad_corpus.json"
	processed_questions_path: Path = project_root / "data" / "squad_questions.json"
	processed_chunks_path: Path = project_root / "data" / "squad_chunks.json"
	processed_chunk_embeddings_path: Path = project_root / "data" / "squad_chunk_embeddings.npy"
	model_cache_dir: Path = project_root / "models"
	output_path: Path = project_root / "outputs" / "batch_predictions.jsonl"
	dataset_name: str = "squad"
	dataset_corpus_split: str = "train"
	dataset_question_split: str = "validation"
	dataset_context_field: str = "context"
	dataset_question_field: str = "question"
	dataset_corpus_limit: int = 5000
	dataset_question_limit: int = 50

	# Retrieval mode: semantic retrieval + reranking over chunked passages.
	retrieval_mode: str = "semantic-hybrid"
	embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	# Tuned QA reader model from Hugging Face with a practical download size.
	generator_model_name: str = "deepset/minilm-uncased-squad2"
	chunk_size_words: int = 80
	chunk_overlap_words: int = 20
	retrieval_candidate_k: int = 15
	rerank_semantic_weight: float = 0.8
	rerank_lexical_weight: float = 0.2

	top_k: int = 3
	max_new_tokens: int = 128
	min_documents: int = 3


CONFIG = RAGConfig()
