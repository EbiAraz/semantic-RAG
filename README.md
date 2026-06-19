---
title: Semantic RAG
emoji: "🔎"
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# Semantic RAG

This project is a lightweight Retrieval-Augmented Generation demo.

## What it does

- Downloads a Hugging Face QA dataset instead of using only hand-written documents.
- Builds chunked semantic retrieval over the downloaded corpus.
- Uses hybrid retrieval with embedding search plus lexical reranking.
- Uses a tuned Hugging Face question-answering model to answer from retrieved context.

## Main files

- `src/data-loader.py`: loads the dataset corpus and batch questions.
- `src/inference.py`: retrieval + answer generation pipeline.
- `src/preprocess.py`: builds reusable corpus, chunk, and embedding caches.
- `src/main.py`: interactive CLI.
- `src/batch-predict.py`: batch prediction runner.
- `src/evaluate-model.py`: computes EM and F1 over the validation set.
- `src/streamlit_app.py`: Streamlit UI implementation.
- `app.py`: Hugging Face Spaces and local Streamlit entrypoint.
- `run_pipeline.py`: top-level orchestration for preprocess, batch, and evaluate.

## Run

```powershell
python src/preprocess.py
```

```powershell
python src/main.py
```

```powershell
python src/batch-predict.py
```

```powershell
python src/evaluate-model.py --limit 25
```

```powershell
python src/evaluate-model.py --fast
```

```powershell
python src/smoke-test.py
```

```powershell
streamlit run src/streamlit_app.py
```

```powershell
python app.py
```

For Hugging Face Spaces, use the repository root `app.py` as the Streamlit entrypoint and `requirements.txt` for dependencies.

```powershell
python run_pipeline.py
```

```powershell
python run_pipeline.py --fast-evaluate
```

## Notes

- The first run will download the SQuAD dataset and the Hugging Face QA model. The default QA reader is tuned for accuracy and sized to be practical to run locally.
- Run preprocessing first if you want reusable local caches in `data/squad_corpus.json`, `data/squad_questions.json`, `data/squad_chunks.json`, and `data/squad_chunk_embeddings.npy`.
- Outputs from batch mode are written to `outputs/batch_predictions.jsonl`.
- Evaluation reports are written to `outputs/evaluation_report.json`.
- `python src/evaluate-model.py --fast` is the quickest way to sanity-check quality after retrieval changes.
- `python src/smoke-test.py` runs a practical end-to-end check with factual, general, and life-style prompts.
- The Streamlit app provides a browser UI for asking questions and inspecting retrieved context.
- `python app.py` is a convenient root-level launcher for the Streamlit UI.
- `python run_pipeline.py` runs preprocessing, batch prediction, and evaluation in sequence.

## Hugging Face Spaces

1. Create a new Space with SDK set to `Streamlit`.
2. Set the app file to `app.py` at the repository root.
3. Keep `requirements.txt` at the root so Spaces installs the correct dependencies.
4. Push the repo without local-only folders such as `.venv/`, `models/`, `outputs/`, and `results/`.
5. The repository already includes `.gitignore` and `.hfignore` so generated caches and local artifacts stay out of the deployed Space by default.
6. On first boot, the Space will download the QA and embedding models, then build runtime caches as needed.
7. If you want faster cold starts, run `python src/preprocess.py` before upload and then decide explicitly whether to include generated `data/` caches in the repo.

### Recommended Space Settings

- SDK: `Streamlit`
- App file: `app.py`
- Python version: `3.10` or newer
- Hardware: `CPU Basic` is enough to boot, but model loading and evaluation will be slow
- Persistent storage: optional, but useful if you want model and retrieval caches to survive restarts

### Deploy-safe Files

Safe to keep in the repo:

- `app.py`
- `requirements.txt`
- `README.md`
- `src/`
- `data/documents.txt`
- `data/questions.txt`

Ignored by default for deployment:

- `.venv/`
- `models/`
- `outputs/`
- `results/`
- generated cache files under `data/`
