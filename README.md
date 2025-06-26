# 🤖 RAG-Chatbot

A **Retrieval-Augmented Generation (RAG)** based chatbot that enhances Large Language Models (LLMs) with **document-aware responses**. Users can upload documents and interact with an intelligent assistant that uses context-based reasoning from those documents.

---

## 🚀 Features

- 📄 **Multi-format document upload** (PDF, HTML, URLs, etc.)
- 🔍 **Context-aware answers** using vector search and LLMs
- ⚡ **FastAPI backend** for modular and scalable APIs
- 💬 Interactive **chat interface** powered by Streamlit
- 📚 Chunking and embedding via **LangChain**
- 🔐 Environment variable management via `.env`


---

## 📦 Tech Stack

| Layer           | Technology                             |
|----------------|-----------------------------------------|
| 🧠 LLM          | OpenAI / HuggingFace Transformers       |
| 🧾 API Backend  | FastAPI                                |
| 🔎 Vector Store | FAISS                                  |
| 📄 Loaders      | LangChain (PDF, HTML, Web)             |
| 💬 Frontend     | Streamlit                              |


---

## 📁 Folder Structure

RAG-Chatbot/
│
├── App/ # Streamlit app interface
├── Modules/ # RAG core modules
│ └── RAG/ # Document processing and retrieval
├── api/ # Optional backend API endpoints
├── .venv/ # uv virtual environment (not pushed)
├── .gitignore
├── requirements.txt
├── README.md

## 🔌 Running the FastAPI Backend

The project includes a modular FastAPI server for backend APIs (chat endpoint, document upload, history, etc.).

### ▶️ Run FastAPI with Uvicorn

```bash
uvicorn api.main:app --reload
```

![Screenshot 2025-06-24 162436](https://github.com/user-attachments/assets/9592ca01-a113-42c8-bcfc-adacc39ae9c0)




uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

