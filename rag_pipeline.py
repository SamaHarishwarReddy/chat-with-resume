from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os


def build_qa_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it in your Hugging Face Space Secrets.")

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def qa_chain(question):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        prompt = f"""You are a helpful assistant. Use ONLY the following resume content to answer the question. If the answer is not in the resume, say "I don't have that information in this resume."

Resume content:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        return response.content

    return qa_chain
