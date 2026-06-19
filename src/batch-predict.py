import json
import importlib.util
from pathlib import Path

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

def main() -> None:
	data_loader = _load_data_loader_module()
	try:
		questions = data_loader.load_questions()
	except Exception:
		questions = []

	if not questions:
		questions_path = CONFIG.data_path.with_name("questions.txt")
		if questions_path.exists():
			questions = [line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]

	if not questions:
		print("No questions found in the downloaded dataset or fallback file.")
		return

	engine = RAGEngine()
	CONFIG.output_path.parent.mkdir(parents=True, exist_ok=True)

	with CONFIG.output_path.open("w", encoding="utf-8") as f:
		for question in questions:
			result = engine.answer(question)
			f.write(json.dumps(result, ensure_ascii=False) + "\n")

	print(f"Saved {len(questions)} predictions to {CONFIG.output_path}")


if __name__ == "__main__":
	main()
