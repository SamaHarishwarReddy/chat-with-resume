from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os


def build_qa_chain(pdf_path):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed & store in FAISS (runs locally, no API cost)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Setup LLM via Groq (free, fast LLaMA 3)
    # API key is loaded from .env file — never hardcode secrets
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # 5. Build QA chain using new LangChain style (no deprecated RetrievalQA)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def qa_chain(question):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        prompt = f"""You are a helpful assistant. Use the following resume content to answer the question.

Resume content:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        return response.content

    return qa_chain
