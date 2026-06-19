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

	sample_questions = get_sample_questions()

	with st.sidebar:
		st.subheader("Model Info")
		st.write(f"Dataset: {CONFIG.dataset_name}")
		st.write(f"Reader: {CONFIG.generator_model_name}")
		st.write(f"Retrieval mode: {CONFIG.retrieval_mode}")
		st.write(f"Default top_k: {CONFIG.top_k}")
		st.divider()
		st.subheader("Repo Links")
		st.markdown("- [GitHub Repository](https://github.com/EbiAraz/semantic-RAG)")
		st.markdown("- [Hugging Face Space](https://huggingface.co/spaces/EbiAraz/semantic-rag)")
		st.markdown("- [README](https://github.com/EbiAraz/semantic-RAG/blob/main/README.md)")
		st.markdown("- [app.py](https://github.com/EbiAraz/semantic-RAG/blob/main/app.py)")
		st.markdown("- [src/inference.py](https://github.com/EbiAraz/semantic-RAG/blob/main/src/inference.py)")
		st.markdown("- [src/streamlit_app.py](https://github.com/EbiAraz/semantic-RAG/blob/main/src/streamlit_app.py)")
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

	with st.expander("Quick Validation", expanded=False):
		st.caption("Run a small built-in check of factual and life-style questions.")
		if st.button("Run quick validation", use_container_width=True):
			with st.spinner("Running quick validation..."):
				engine = get_engine()
				quick_questions = [
					"What does RAG stand for?",
					"What is FAISS used for?",
					"How can I improve sleep quality?",
				]
				quick_results = [engine.answer(q, top_k=top_k) for q in quick_questions]
			for idx, quick in enumerate(quick_results, start=1):
				st.markdown(f"**{idx}. {quick['question']}**")
				st.write(quick["answer"])
				st.caption(f"intent={quick.get('query_intent')}, score={quick.get('answer_score', 0.0):.4f}")

	if submit and question.strip():
		st.session_state["question"] = question.strip()
		with st.spinner("Loading models and building retrieval cache (first run can take several minutes)..."):
			try:
				engine = get_engine()
			except Exception as exc:
				st.error("Engine initialization failed. Check Space logs for details.")
				st.exception(exc)
				return

		result = engine.answer(question.strip(), top_k=top_k)

		st.subheader("Answer")
		st.write(result["answer"] or "No answer returned.")
		col1, col2 = st.columns([1, 1])
		with col1:
			st.metric("Confidence", f"{result['answer_score']:.4f}")
		with col2:
			source_idx = result.get("answer_source_index", -1)
			if source_idx >= 0:
				st.info(f"Source: Retrieved chunk #{source_idx + 1}")
			else:
				st.warning("Low confidence - no reliable source")

		intent_col, source_col = st.columns([1, 1])
		with intent_col:
			st.caption(f"Intent: {result.get('query_intent', 'general')}")
			st.caption(f"Min confidence: {result.get('min_confidence', 0.0):.3f}")
			st.caption(f"Answer mode: {result.get('answer_mode', 'retrieval')}")
		with source_col:
			squad_count = sum(1 for item in result["context"] if item.get("source") == "squad")
			local_count = sum(1 for item in result["context"] if item.get("source") == "local")
			st.caption(f"Top-{len(result['context'])} sources: squad={squad_count}, local={local_count}")

		st.subheader("Retrieved Context")
		for idx, item in enumerate(result["context"]):
			with st.expander(f"Chunk #{idx + 1} - Score {item['score']:.4f}"):
				st.caption(f"source={item.get('source', 'unknown')}, bonus={item.get('intent_bonus', 0.0):.4f}")
				st.write(item["document"])

		st.subheader("Debug Output")
		st.json(result)
	elif not question.strip():
		st.info("Type a question or choose one of the examples in the sidebar.")
		st.caption("The model loads on first question. Initial startup may take a few minutes on CPU Basic.")


if __name__ == "__main__":
	main()
