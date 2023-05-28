#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Androse_Jes
#
# Created:     21-05-2023
# Copyright:   (c) Androse_Jes 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os, time, pinecone
from apikey import openai_apikey, pinecone_apikey, pinecone_env
from langchain.embeddings import HuggingFaceInstructEmbeddings

#licence setup
os.environ['OPENAI_API_KEY'] = openai_apikey
os.environ['PINECONE_API_KEY'] = pinecone_apikey
os.environ['PINECONE_API_ENV'] = pinecone_env

#setup
#embeddings = OpenAIEmbeddings(openai_api_key=openai_apikey)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",model_kwargs={"device": "cuda"})

#variable
ask = False
refresh = False

#init pinecone
pinecone.init(
    api_key=pinecone_apikey,  # find at app.pinecone.io
    environment=pinecone_env  # next to api key in console
)

#init openai
llm = OpenAI(temperature=0, openai_api_key=openai_apikey)
chain = load_qa_chain(llm, chain_type="stuff")

#Database
indexlist = pinecone.list_indexes()
index = indexlist[0]

#heading
st.subheader("Custom PDF Search!!!")
st.session_state.horizontal = True
options = ["existing database","update existing database","new database"]
choice = st.radio("Choose anyone option",options,horizontal=st.session_state.horizontal,index=0)

if choice == options[0]:
    st.write(f"Available Database: '{index}'")
    docsearch = Pinecone.from_texts("", embeddings, index_name=index)
    ask = True


elif choice == options[1]:
    ask = False
    uploaded_files = st.file_uploader(f"Choose PDF file(s) to update the exiting database:'{index}'", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            # Access the file attributes
            filename = file.name
            # Create an instance of the UnstructuredPDFLoader class for each file
            loader = UnstructuredPDFLoader(filename)
            # Load the PDF file
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index)
        ask = True
    else:
        st.write("No files uploaded.")

else:
    ask = False
    new_index = st.text_input(f"Make a new database by deleting existing: '{index}'")
    if new_index:
        with st.spinner(f'please wait exiting database: "{index}"  is getting deleting...'):
            pinecone.delete_index(index)
        with st.spinner(f'please wait new database "{new_index}" is creating...'):
            pinecone.create_index(new_index, dimension=1536)
            #time.sleep(10)
            st.success('Done!')
            uploaded_files = st.file_uploader("Choose a PDF file(s)", accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    # Access the file attributes
                    filename = file.name
                    # Create an instance of the UnstructuredPDFLoader class for each file
                    loader = UnstructuredPDFLoader(filename)
                    # Load the PDF file
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
                    texts = text_splitter.split_documents(data)
                    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index)
                    ask=False
                    refresh = True

            else:
                st.write("No files uploaded.")

if ask:
    prompt = st.text_input("Ask any question?")
    if prompt:
        docs = docsearch.similarity_search(prompt)
        answer = chain.run(input_documents=docs, question=prompt)
        st.write(answer)
elif refresh:
    with st.spinner("restarting app please wait..."):
        time.sleep(2)
        st.experimental.rerun()









