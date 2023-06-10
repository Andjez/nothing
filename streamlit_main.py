#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Androse_Jes
#
# Created:     28-05-2023
# Copyright:   (c) Androse_Jes 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os, re
import streamlit as st
from langchain import OpenAI
from apikey import openai_apikey
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "source" not in st.session_state:
    st.session_state["source"] = []
    
if "time_sec" not in st.session_state:
    st.session_state["time_sec"] = []

os.environ['OPENAI_API_KEY'] = openai_apikey

#embedding
#instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})


@st.cache_resource
def load_chain(yt_link):
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print("1")
    store = FAISS.from_documents(documents=texts,embedding=instructor_embeddings)
    print("2")
    retriever = store.as_retriever(search_kwargs={"k": 3})
    print("3")
    return retriever

yt_link = st.text_input("enter youtube link here!")

if yt_link:
    chain = load_chain(yt_link)

def get_text():
    input_text = st.text_input("You: ", "")
    return input_text

user_input = get_text()

if user_input:
    #result = chain({"question": user_input})
    #output = f"Answer: {result['answer']}\nSources: {result['sources']}"
    docs = chain.get_relevant_documents(user_input)
    output = docs[0].page_content
    source_01 = docs[0].metadata.get('source')
    time_sec_01 =docs[0].metadata.get('length')
     
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.source.append(source_01)
    st.session_state.time_sec.append(time_sec_01)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["source"][i], key=str(i+99))
        message(st.session_state["time_sec"][i], key=str(i+999))
        message(st.session_state["past"][i], is_user=True, key=str(i+9999) + "_user")
