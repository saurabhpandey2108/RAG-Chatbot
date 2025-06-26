# 🤖 RAG-Chatbot

A **Retrieval-Augmented Generation (RAG)** based chatbot that enhances Large Language Models (LLMs) with **document-aware responses**. Users can upload documents and interact with an intelligent assistant that uses context-based reasoning from those documents.

---

## 🚀 Features

- 📄 **Multi-format document upload** (PDF, HTML, URLs, etc.)
- 🔍 **Context-aware answers** using vector search and LLMs
- 💬 Interactive **chat interface** powered by Streamlit
- ⚡ Fast semantic retrieval using **FAISS**
- 📚 Chunking and embedding via **LangChain**
- 🔐 Environment variable management via `.env`

---

## 📦 Tech Stack

| Layer           | Technology                             |
|----------------|-----------------------------------------|
| 🧠 LLM          | OpenAI / HuggingFace Transformers       |
| 📄 Document Loaders | LangChain loaders (`PyPDFLoader`, `WebBaseLoader`) |
| 🔎 Vector Store | FAISS                                  |
| 💬 Frontend     | Streamlit                              |
| 🧰 Backend Utils| Python, dotenv, tempfile                |

---

## 📁 Folder Structure

