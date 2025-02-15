# https://www.youtube.com/watch?v=MoqgmWV1fm8

import streamlit as st

from bs4 import BeautifulSoup
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import GooglePalm
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as palm
import time
from pathlib import Path

dashboard = st.sidebar.selectbox("select RAG tool",["link RAG","pdf RAG"])

st.title("Akj Advance News Research Tool 📈")
st.sidebar.title("News Article URLs")
links = st.sidebar.number_input("Insert a number of url links")
if links:
    urls = []
    try:
        for i in range(links):
            url = st.sidebar.text_input(f"URL {i+1}")
            urls.append(url)
    except:
        for i in range(1):
            url = st.sidebar.text_input(f"URL {i+1}")
            urls.append(url)
        
    
    process_url_clicked = st.sidebar.button("Submit URLs")
    
    # GOOGLE_API_KEY = st.secrets.API_KEY
    groq_key = st.secrets.API_KEY
    llm = ChatGroq(
    temperature=0,
    groq_api_key = key,
    model_name = 'llama-3.3-70b-versatile')
    
    palm.configure(api_key = GOOGLE_API_KEY)
    models = [m for m in palm.list_models() 
              if 'generateText' 
              in m.supported_generation_methods]
    
    model_bison = models[0]
    from google.api_core import retry
    @retry.Retry()
    def generate_text(prompt,
                      model=model_bison,
                      temperature=0.0):
        return palm.generate_text(prompt=prompt,
                                  model=model,
                                  temperature=temperature)
    
    
    query = st.text_input("Question: ") 

    if process_url_clicked:
       
        # load data
        loader = WebBaseLoader(url)
        st.text("Data Loading...Started...✅✅✅")
        data = loader.load().pop().page_content
        # split data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 0, length_function = len, add_start_index = True,)
        st.text("Text Splitter...Started...✅✅✅")
        # docs = text_splitter.split_documents(data)
        docs= text_splitter.create_documents(data)
        # create embeddings and save it to FAISS index
        embeddings = HuggingFaceEmbeddings()
        # GooglePalmEmbeddings(google_api_key=GOOGLE_API_KEY)
        vectorstore_palm = FAISS.from_documents(docs, embeddings)
        st.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
    
        # Save the vectorstore object locally
        vectorstore_palm.save_local("vectorstore")   
        # Load the vectorstore object
        x = FAISS.load_local("vectorstore", embeddings,allow_dangerous_deserialization=True)
    
        if dashboard=="link RAG":
            chain = RetrievalQA.from_chain_type(llm =llm,
                chain_type="stuff",
                retriever=x.as_retriever(),
                input_key ="query",
                return_source_documents=True)
            
            res = chain.invoke(query)
    
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            
            st.header("Answer")
            st.header(query)
            st.write(res['result'])
            st.write(res['source_documents'])
        if dashboard=="pdf RAG":
            # Run similarity search query
            # q = "What are the 3 main perspectives regarding zero carbon shipping?"
            v = vectorstore_palm.similarity_search(query, include_metadata=True)
    
            # Run the chain by passing the output of the similarity search
            chain = load_qa_chain(llm, chain_type="stuff")
            res = chain({"input_documents": v, "question": query})
            st.write(res["output_text"])
            st.write(res["input_documents"]) 
    
    task_list = ["Write","Short","Debug","Documentation","Translate","Summary"]
    task = st.selectbox("What is your task",task_list)
    input = st.text_area("ask your question")
    if st.button("Submit"):
        with st.spinner("processing"):
            if task == "Write": 
                prompt_template = """
                            {priming}
    
                            {question}
    
                            {decorator}
    
                            Your solution:
                            """
                priming_text = "You are an expert at critical reasoning and writing blog from holisitc point of view."
                decorator = "Keep the tone friendly and compelling"
    
                completion = generate_text(prompt = prompt_template.format(priming=priming_text,
                                question=input,
                                decorator=decorator))
                output = completion.result
                st.markdown(output)
    
            if task == "Short":
                st.subheader("Short description!")
                prompt_template = """
                Your task is to help a marketing team create a description for a blog website
                at most 50 words.
                {question}
                Also provide 5 keywords for SEO marketting.
                """
                completion = generate_text(prompt = prompt_template.format(question=input))
                output = completion.result
                st.markdown(output)
    
            if task == "Debug":
                st.subheader("The Debugger mode is activated")
                prompt_template = """
                Check the sentence if it has any grammar mistake and if yes then provide the flaws
    
                {question}
    
                Explain in detail what you found and why it 
                """
                completion = generate_text(prompt = prompt_template.format(question=input),
                                            temperature=0.5)
                output = completion.result
                st.markdown(output)
    
            if task == "Documentation":
                st.subheader("Lets write up the documnet")
                prompt_template = """
                Please write technical documentation for this code and \n
                make it easy for a non developer to understand:
    
                {question}
    
                Output the results in markdown
                """
                completion = generate_text(prompt = prompt_template.format(question=input))
                output = completion.result
                st.markdown(output)
    
            if task == "Translate":
                st.subheader("Explore multiple ways to write!")
                prompt_template = """
                {priming}
    
                {question}
    
                {decorator}
    
                Your solution:
                """
                priming_text = """
                You are expert linguist and know french
                """
                decorator = "Keep the tone friendly and compelling"
                completion = generate_text(prompt = prompt_template.format(priming=priming_text,
                        question=input,
                        decorator=decorator))
                output = completion.result
                st.markdown(output)
    
            if task == "Summary":
                st.subheader("The most easy way to do it is shown below")
                prompt_template = """
                {priming}
    
                {question}
    
                {decorator}
    
                Your solution:
                """
                priming_text = """
                Your task is to generate a short summary in at most 50 words.
                Focus on any aspect that are business oriented.
                """
                decorator = "Keep the tone friendly and compelling"
                completion = generate_text(prompt = prompt_template.format(priming=priming_text,
                        question=input,
                        decorator=decorator))
                output = completion.result
                st.markdown(output)
