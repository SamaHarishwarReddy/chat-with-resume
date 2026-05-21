import streamlit as st
from rag_pipeline import build_qa_chain
import os
import tempfile

st.set_page_config(page_title="Chat with Resume", page_icon="📄")

st.title("📄 Chat with Your Resume")
st.caption("Upload a resume PDF and ask questions about it using AI")

with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Upload a resume PDF
    2. Wait for it to process
    3. Ask questions like:
       - *"What skills does this person have?"*
       - *"Are they a good fit for an ML role?"*
       - *"What projects have they built?"*
       - *"What is their GPA?"*
    """)
    st.divider()
    st.caption("Powered by LangChain + FAISS + LLaMA 3")

uploaded = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    with st.spinner("Reading and indexing resume..."):
        try:
            qa = build_qa_chain(tmp_path)
            st.success("Resume loaded! Ask me anything below.")
        except Exception as e:
            st.error(f"Error loading resume: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the resume...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = qa(question)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

    os.unlink(tmp_path)

else:
    st.info("Please upload a PDF resume to get started.")
