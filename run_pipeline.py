from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(title: str, command: list[str], project_root: Path) -> None:
	print(f"\n== {title} ==")
	subprocess.run(command, check=True, cwd=project_root)


def main() -> None:
	project_root = Path(__file__).resolve().parent
	python_executable = sys.executable

	parser = argparse.ArgumentParser(description="Run the full Semantic RAG pipeline.")
	parser.add_argument("--skip-preprocess", action="store_true", help="Skip dataset preprocessing.")
	parser.add_argument("--skip-batch", action="store_true", help="Skip batch prediction.")
	parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation.")
	parser.add_argument("--evaluate-limit", type=int, default=25, help="Number of examples to score during evaluation.")
	parser.add_argument("--fast-evaluate", action="store_true", help="Use the evaluator's fast mode with fewer examples and one retrieved passage.")
	args = parser.parse_args()

	if not args.skip_preprocess:
		_run_step(
			"Preprocess Dataset",
			[python_executable, "src/preprocess.py"],
			project_root,
		)

	if not args.skip_batch:
		_run_step(
			"Batch Predictions",
			[python_executable, "src/batch-predict.py"],
			project_root,
		)

	if not args.skip_evaluate:
		evaluate_command = [python_executable, "src/evaluate-model.py", "--limit", str(args.evaluate_limit)]
		if args.fast_evaluate:
			evaluate_command.append("--fast")
		_run_step(
			"Evaluate Model",
			evaluate_command,
			project_root,
		)

	print("\nPipeline complete.")


if __name__ == "__main__":
	main()
