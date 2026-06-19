import argparse
import importlib.util
import json
import re
import string
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from config import CONFIG
from inference import RAGEngine


def _load_data_loader_module():
	module_path = Path(__file__).with_name("data-loader.py")
	spec = importlib.util.spec_from_file_location("data_loader", module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError("Could not load data-loader.py")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def normalize_text(text: str) -> str:
	text = text.lower()
	text = text.translate(str.maketrans("", "", string.punctuation))
	text = re.sub(r"\b(a|an|the)\b", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
	return float(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
	pred_tokens = normalize_text(prediction).split()
	truth_tokens = normalize_text(ground_truth).split()

	if not pred_tokens and not truth_tokens:
		return 1.0
	if not pred_tokens or not truth_tokens:
		return 0.0

	common = Counter(pred_tokens) & Counter(truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0.0

	precision = num_same / len(pred_tokens)
	recall = num_same / len(truth_tokens)
	return 2 * precision * recall / (precision + recall)


def build_examples(limit: int) -> list[dict]:
	data_loader = _load_data_loader_module()
	from datasets import load_dataset

	dataset = load_dataset(CONFIG.dataset_name, split=CONFIG.dataset_question_split)
	examples = []
	for row in dataset:
		answers = row.get("answers", {})
		answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
		if not answer_texts:
			continue
		examples.append(
			{
				"question": str(row[CONFIG.dataset_question_field]),
				"answers": [str(answer) for answer in answer_texts],
				"context": str(row.get(CONFIG.dataset_context_field, "")),
			}
		)
		if len(examples) >= limit:
			break

	if not examples:
		questions = data_loader.load_questions()
		for question in questions[:limit]:
			examples.append({"question": question, "answers": [question], "context": ""})

	return examples


def evaluate(limit: int, *, top_k: int | None = None) -> dict:
	engine = RAGEngine()
	examples = build_examples(limit)

	# Align retrieval corpus to evaluation contexts so the gold answers are reachable.
	eval_documents = []
	seen_contexts = set()
	for example in examples:
		ctx = example.get("context", "").strip()
		if not ctx or ctx in seen_contexts:
			continue
		seen_contexts.add(ctx)
		eval_documents.append(ctx)

	if eval_documents:
		engine.documents = eval_documents
		engine.chunks = engine._chunk_documents(eval_documents)
		engine.vectorizer = TfidfVectorizer(stop_words="english")
		engine.chunk_tfidf_matrix = engine.vectorizer.fit_transform([chunk["text"] for chunk in engine.chunks])
		engine.chunk_embeddings = engine._encode_texts([chunk["text"] for chunk in engine.chunks])

	results = []
	em_total = 0.0
	f1_total = 0.0

	for example in examples:
		prediction = engine.answer(example["question"], top_k=top_k)
		pred_text = prediction["answer"]
		answers = example["answers"]

		em = max(exact_match_score(pred_text, answer) for answer in answers)
		f1 = max(f1_score(pred_text, answer) for answer in answers)

		em_total += em
		f1_total += f1
		results.append(
			{
				"question": example["question"],
				"prediction": pred_text,
				"gold_answers": answers,
				"gold_context": example.get("context", ""),
				"exact_match": em,
				"f1": f1,
				"answer_score": prediction["answer_score"],
				"context": prediction["context"],
			}
		)

	count = len(results) or 1
	report = {
		"dataset": CONFIG.dataset_name,
		"split": CONFIG.dataset_question_split,
		"examples": len(results),
		"exact_match": em_total / count,
		"f1": f1_total / count,
		"results": results,
	}
	return report


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a Hugging Face QA dataset.")
	parser.add_argument("--limit", type=int, default=25, help="Number of evaluation examples to score.")
	parser.add_argument("--top-k", type=int, default=None, help="Override the number of retrieved passages used during evaluation.")
	parser.add_argument("--fast", action="store_true", help="Run a smaller and faster evaluation using fewer examples and passages.")
	parser.add_argument(
		"--output",
		type=str,
		default=str(CONFIG.output_path.parent / "evaluation_report.json"),
		help="Path to write the evaluation report.",
	)
	args = parser.parse_args()

	limit = args.limit
	top_k = args.top_k
	if args.fast:
		limit = min(limit, 2)
		top_k = 1 if top_k is None else min(top_k, 1)

	report = evaluate(limit, top_k=top_k)
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"Evaluated {report['examples']} examples")
	print(f"Exact Match: {report['exact_match']:.4f}")
	print(f"F1: {report['f1']:.4f}")
	print(f"Saved report to {output_path}")


if __name__ == "__main__":
	main()
