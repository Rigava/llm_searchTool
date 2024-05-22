# https://www.youtube.com/watch?v=MoqgmWV1fm8

import streamlit as st
import os
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.llms import GooglePalm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

import time

st.title("Akj News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
GOOGLE_API_KEY = st.secrets.key
llm = GooglePalm(google_api_key=GOOGLE_API_KEY)

query = main_placeholder.text_input("Question: ") 

if process_url_clicked:
   
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = GooglePalmEmbeddings(google_api_key=GOOGLE_API_KEY)
    vectorstore_palm = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the vectorstore object locally
    vectorstore_palm.save_local("vectorstore")   
    # Load the vectorstore object
    x = FAISS.load_local("vectorstore", embeddings,allow_dangerous_deserialization=True)

    chain = RetrievalQA.from_chain_type(llm =llm,
        chain_type="stuff",
        retriever=x.as_retriever(),
        input_key ="query",
        return_source_documents=True)
    # res = chain("What is the pooling concept?")
    res = chain(query)

    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(res['result'])
    st.write(res['source_documents'])

