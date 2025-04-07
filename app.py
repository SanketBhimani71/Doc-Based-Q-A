import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredFileLoader

# Load environment variables
load_dotenv()

st.title('📄 Q&A Chatbot for Documents')

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your document (PDF, Markdown, TXT, DOCX)",
    type=["pdf", "md", "txt", "docx"]
)

def file_upload():
    try:
        # Create temp dir and save uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.sidebar.success("✅ File uploaded successfully!")

        # Use UnstructuredFileLoader for better compatibility
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

        st.sidebar.success("✅ Document loaded successfully!")
        return documents
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")
        return None

def embed_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=filter_complex_metadata(chunks),
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )

    return vectorstore

# If user uploaded file
if uploaded_file:
    documents = file_upload()

    if documents:
        # Build Vectorstore
        vectorstore = embed_documents(documents)

        # Set up retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Set up LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=500
        )

        # Prompt Template
        system_prompt = """You are an assistant for question-answering tasks.
                            Use the retrieved context below to answer the user's question.
                            If you don't know the answer, say so clearly.
                            Keep answers concise and under 3 sentences.
                            
                            {context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Chat input
        user_query = st.chat_input("Ask a question about the document...")

        if user_query:
            with st.spinner("Searching for answer..."):
                result = rag_chain.invoke({"input": user_query})
                st.write(result["answer"])
