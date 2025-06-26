import tempfile
import streamlit as st
from dotenv import load_dotenv
import os
import torch
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Changed to FAISS for better performance
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
import time

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Page configuration with custom theme
st.set_page_config(
    page_title="Enhanced RAG ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for system stats
with st.sidebar:
    st.title("System Stats 📊")
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
    
    with st.expander("About"):
        st.write("This is an RAG system supporting URL, PDF, and HTML inputs.")

# Main title with colored header
colored_header(label="RAG Question-Answering System", description="Made by SAURABH PANDEY", color_name="green-70")

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("🔑 OPENAI_API_KEY is not set in the environment variables")
    st.stop()

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Input mode selection with enhanced UI
input_mode = st.radio(
    "Select Input Mode 📥",
    ["Website URL", "File Upload", "HTML Content"],
    horizontal=True,
    help="Choose your input source"
)

documents = []
start_time = time.time()

if input_mode == "Website URL":
    url = st.text_input(
        "Enter a URL 🌐",
        placeholder="https://example.com"
    )
    
    if st.button("Initialize RAG System from URL 🚀"):
        with st.spinner("Loading and processing documents from URL..."):
            if url:
                loader = WebBaseLoader(url)
                documents.extend(loader.load())
            else:
                st.error("Please provide a URL.")
                st.stop()

elif input_mode == "File Upload":
    uploaded_file = st.file_uploader("Upload a file 📄", type=["txt", "pdf"])
    
    if st.button("Initialize RAG System from File 🚀"):
        with st.spinner("Loading and processing documents from file..."):
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1]
                if file_type == 'txt':
                    documents.append(uploaded_file.read().decode("utf-8"))
                elif file_type == 'pdf':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    try:
                        loader = PyPDFLoader(temp_path)
                        documents.extend(loader.load())
                    finally:
                        os.unlink(temp_path)
            else:
                st.error("Please upload a file.")
                st.stop()

else:  # HTML Content
    html_content = st.text_area(
        "Paste your HTML content 📝",
        height=200,
        help="Paste your HTML content here"
    )
    
    if st.button("Initialize RAG System from HTML 🚀"):
        with st.spinner("Processing HTML content..."):
            if html_content:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as tmp_file:
                    tmp_file.write(html_content)
                    temp_path = tmp_file.name
                try:
                    loader = BSHTMLLoader(temp_path)
                    documents.extend(loader.load())
                finally:
                    os.unlink(temp_path)
            else:
                st.error("Please provide HTML content.")
                st.stop()

if documents:
    with st.spinner("🔄 Processing documents..."):
        # Optimized text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for better context
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Using sentence-transformers for embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'}
        )
        
        # Using FAISS for better performance
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        st.session_state.processing_time = time.time() - start_time
        st.success("✅ RAG system initialized successfully!")

if st.session_state.vectorstore is not None:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable assistant specialized in analyzing documents. Provide detailed, accurate answers based on the context. If uncertain, clearly state 'I don't know' rather than speculating."),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    chain = prompt | llm
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Ask a Question 💭")
        question = st.text_area(
            "What would you like to know?",
            placeholder="Enter your question here..."
        )
        
        if st.button("Get Answer 🔍"):
            if question:
                with st.spinner("🤔 Thinking..."):
                    start_time = time.time()
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}  # Optimized number of documents
                    )
                    docs = retriever.invoke(question)
                    
                    response = chain.invoke({
                        "question": question,
                        "context": docs
                    })
                    
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
                    st.session_state.processing_time = time.time() - start_time
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.subheader("Answer 📝")
        if 'last_response' in st.session_state:
            st.markdown(f"""<div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                {st.session_state.last_response}
            </div>""", unsafe_allow_html=True)
            
            with st.expander("📚 Show Retrieved Context"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**📄 Relevant Document {i}:**")
                    st.markdown(f"""<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                        {doc.page_content}
                    </div>""", unsafe_allow_html=True)
                    add_vertical_space(1)

else:
    st.info("🚀 Please initialize the RAG system by providing input and clicking the Initialize button")