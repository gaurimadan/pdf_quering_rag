import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time


load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("PDF Quering")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def process_pdf(uploaded_file):
    """Extract text from PDF and create vector embeddings"""
    if uploaded_file is not None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

       
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

       
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("PDF Processed & Vector Store Created!")


if uploaded_file:
    process_pdf(uploaded_file)


user_query = st.text_input("What's your query")


if user_query and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_query})
    st.write("Response Time:", time.process_time() - start)
    st.write(response['answer'])

    