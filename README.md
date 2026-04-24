# Chat with Your Resume — RAG-powered Q&A Bot

Ask AI questions about any resume PDF using LangChain, FAISS, and LLaMA 3.

## Tech Stack
- **LangChain** — RAG orchestration
- **FAISS** — local vector database for semantic search
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` (runs locally, free)
- **Groq + LLaMA 3** — fast, free LLM inference
- **Streamlit** — frontend UI

## Setup

### 1. Clone and install
```bash
git clone <your-repo>
cd chat-resume
pip install -r requirements.txt
```

### 2. Get a free Groq API key
- Go to https://console.groq.com
- Sign up and create an API key

### 3. Add your API key
```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

### 4. Run locally
```bash
streamlit run app.py
```

## Deploy on Hugging Face Spaces
1. Create a new Space at https://huggingface.co/spaces
2. Choose **Streamlit** as the SDK
3. Push this code to the Space repo
4. Add `GROQ_API_KEY` in Space Settings → Secrets

## Example Questions to Ask
- "What ML skills does this person have?"
- "What projects have they built?"
- "What is their GPA and university?"
- "Are they a good fit for an NLP engineering role?"
- "What certifications do they have?"
