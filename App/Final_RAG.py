import tempfile
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import time

# Initialize session state at the very beginning
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.processing_time = 0.0
    st.session_state.question_history = []
    st.session_state.vectorstore = None
    st.session_state.input_type = "URL"
    st.session_state.last_response = None
    st.session_state.last_context = None

# Custom CSS for professional dark theme
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%); }
    .stButton>button {
        background-color: #00cc66;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: none;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 204, 102, 0.2);
    }
    .stButton>button:hover {
        background-color: #00ff80;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 255, 128, 0.3);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #333333;
        color: #ffffff;
        border: 2px solid #00cc66;
        border-radius: 8px;
    }
    .stMarkdown { color: #ffffff; }
    .stExpander, .stMetric {
        background-color: #333333;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00cc66;
        margin: 10px 0;
    }
    .stRadio>div { 
        background-color: #333333;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00cc66;
    }
    .custom-info-box {
        background-color: #333333;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00cc66;
        margin: 10px 0;
    }
    .custom-answer-box {
        background-color: #333333;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00cc66;
        margin: 10px 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Professional RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with enhanced stats and history
with st.sidebar:
    st.title("System Statistics 📊")
    st.metric(
        "Processing Time",
        f"{st.session_state.processing_time:.2f}s",
        delta=None,
        delta_color="normal"
    )
    
    st.subheader("Recent Questions 📝")
    if st.session_state.question_history:
        for i, (q, a) in enumerate(reversed(st.session_state.question_history[-2:])):
            with st.expander(f"Question {len(st.session_state.question_history)-i}"):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                if st.button("Re-ask", key=f"reask_{i}"):
                    st.session_state.question = q
                    st.rerun()
    
    with st.expander("About System"):
        st.markdown("""
        **Enhanced RAG System Features:**
        - Multi-format input support
        - Real-time processing metrics
        - Question history tracking
        - Context visualization
        - Professional UI/UX
        """)

# Main header
colored_header(
    label="Professional RAG System",
    description="Advanced Question-Answering System",
    color_name="green-70"
)

# Document processing interface
if st.session_state.vectorstore is None:
    st.subheader("Document Processing 📄")
    input_type = st.radio(
        "Select Input Type",
        ["URL", "PDF Upload", "HTML Content"],
        key="input_type"
    )
    
    input_data = None
    if input_type == "URL":
        input_data = st.text_input("Enter URL", placeholder="https://example.com")
        loader_class = WebBaseLoader
    elif input_type == "PDF Upload":
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_data = tmp_file.name
        loader_class = PyPDFLoader
    else:  # HTML Content
        input_data = st.text_area("Paste HTML Content", height=150)
        if input_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                tmp_file.write(input_data.encode())
                input_data = tmp_file.name
        loader_class = BSHTMLLoader

    if st.button("Process Document 🔄", use_container_width=True) and input_data:
        with st.spinner("Processing document..."):
            try:
                start_time = time.time()
                loader = loader_class(input_data)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                splits = text_splitter.split_documents(documents)
                
                embeddings = HuggingFaceEmbeddings()
                st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
                st.session_state.processing_time = time.time() - start_time
                st.success("✅ Document processed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# QA Interface
else:
    # Input type switcher
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.subheader("Ask Questions 💭")
    with col2:
        if st.button("Switch Input Type 🔄", use_container_width=True):
            st.session_state.vectorstore = None
            st.rerun()
    with col3:
        if st.button("Clear History 🗑", use_container_width=True):
            st.session_state.question_history = []
            st.rerun()

    # Question input and processing
    question = st.text_area(
        "What would you like to know?",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="Enter your question here..."
    )
    st.session_state.question = ''

    if st.button("Get Answer 🔍", use_container_width=True) and question:
        with st.spinner("Analyzing..."):
            try:
                start_time = time.time()
                llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a professional assistant specialized in analyzing documents. Provide detailed, accurate answers based on the context. If uncertain, clearly state so."),
                    ("user", "Question: {question}\nContext: {context}")
                ])
                chain = prompt | llm
                
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                response = chain.invoke({"question": question, "context": docs})
                
                st.session_state.question_history.append((question, response.content))
                st.session_state.last_response = response.content
                st.session_state.last_context = docs
                st.session_state.processing_time = time.time() - start_time
                
                # Display answer
                st.markdown("### Answer 📝")
                st.markdown(f"""<div class='custom-answer-box'>{response.content}</div>""", unsafe_allow_html=True)
                
                # Display context
                with st.expander("View Source Context 📚"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"""<div class='custom-info-box'>{doc.page_content}</div>""", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

                