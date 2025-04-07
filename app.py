from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import streamlit as st
import tempfile
import shutil
import os
from langchain_community.vectorstores.utils import filter_complex_metadata
import pysqlite3 as sqlite3

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from chromadb import Client
from chromadb.config import Settings

client = Client(Settings())
collection = client.get_or_create_collection(name="my_collection")


st.title('Q&A Chatbot')

uploaded_file = st.sidebar.file_uploader(
    "Upload your document (PDF, Markdown, TXT, DOCX)", type=["pdf", "md", "txt", "docx"]
)


def file_upload():

    # Create a temporary directory to store uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("File uploaded successfully!")

    loader = UnstructuredLoader(temp_file_path)
    pages = loader.load()
    st.sidebar.success("Document loaded successfully!")
    return pages


def process_file():
    if uploaded_file:

        data = file_upload()
        return data




data = process_file()
load_dotenv()


def embedding_store(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    print("Total number of documents: ", len(docs))


    vectorstore = Chroma.from_documents(
        documents=filter_complex_metadata(docs), embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))

    return vectorstore


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)

system_prompt = """
  "You are an assistant for questions-answering tasks. "
  "Use the following pieces of retrieved context to answer"
  "the question. If you don't know the answer, say that you"
  "don't know. Use three sentences maximum and keep the"
  "answer concise"
  "\n\n"
  "{context}"
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
if data:
    
    vectorstore = embedding_store(data)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10})

    query = st.chat_input('Say something')
    input_query = query

    if input_query:

        question_answer_chain = create_stuff_documents_chain(
            llm, prompt)
        rag_chain = create_retrieval_chain(
            retriever, question_answer_chain)

        response = rag_chain.invoke({"input": input_query})

        st.write(response["answer"])
