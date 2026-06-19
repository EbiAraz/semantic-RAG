import importlib.util
from pathlib import Path

import streamlit as st

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


@st.cache_resource(show_spinner="Loading RAG engine...")
def get_engine() -> RAGEngine:
	return RAGEngine()


@st.cache_data(show_spinner=False)
def get_sample_questions() -> list[str]:
	data_loader = _load_data_loader_module()
	try:
		questions = data_loader.load_questions()
		if questions:
			return questions[:5]
	except Exception:
		pass

	return [
		"What does RAG stand for?",
		"What is FAISS used for?",
		"What city is the capital of Iran?",
	]


def main() -> None:
	st.set_page_config(
		page_title="Semantic RAG",
		page_icon="🔎",
		layout="wide",
	)

	st.title("Semantic RAG")
	st.caption("Ask a question and the app retrieves supporting context from SQuAD before answering.")

	engine = get_engine()
	sample_questions = get_sample_questions()

	with st.sidebar:
		st.subheader("Model Info")
		st.write(f"Dataset: {CONFIG.dataset_name}")
		st.write(f"Reader: {CONFIG.generator_model_name}")
		st.write(f"Retrieval mode: {CONFIG.retrieval_mode}")
		st.write(f"Default top_k: {CONFIG.top_k}")
		st.divider()
		st.subheader("Sample Questions")
		for sample_question in sample_questions:
			if st.button(sample_question, key=f"sample-{sample_question}"):
				st.session_state["question"] = sample_question
				st.rerun()

	question = st.text_area(
		"Enter your question",
		value=st.session_state.get("question", ""),
		height=100,
		placeholder="Ask something about the corpus, e.g. What does RAG stand for?",
	)

	col_left, col_right = st.columns([1, 1])
	with col_left:
		top_k = st.slider("Top retrieved passages", min_value=1, max_value=10, value=CONFIG.top_k)
	with col_right:
		submit = st.button("Ask", type="primary", use_container_width=True)

	if submit and question.strip():
		st.session_state["question"] = question.strip()
		result = engine.answer(question.strip(), top_k=top_k)

		st.subheader("Answer")
		st.write(result["answer"] or "No answer returned.")
		st.metric("Confidence", f"{result['answer_score']:.4f}")

		st.subheader("Retrieved Context")
		for item in result["context"]:
			with st.expander(f"Score {item['score']:.4f}"):
				st.write(item["document"])

		st.subheader("Debug Output")
		st.json(result)
	elif not question.strip():
		st.info("Type a question or choose one of the examples in the sidebar.")


if __name__ == "__main__":
	main()
