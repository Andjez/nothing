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
#from apikey import openai_apikey
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
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "source" not in st.session_state:
    st.session_state["source"] = []
    
if "time_sec" not in st.session_state:
    st.session_state["time_sec"] = []
st.set_page_config(page_title="Youtube Chatbot", page_icon="🃏")
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
user_input = ""
#embedding
#instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
col1,col2 = st.columns(2)
with col1:
    if st.button("Clear Database"):
    # Clears all st.cache_resource caches:
        st.cache_resource.clear()
with col2:
    if st.button("Clear History"):
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["source"] = []
        st.session_state["time_sec"] = []

@st.cache_resource
def load_chain(yt_link):
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    store = FAISS.from_documents(documents=texts,embedding=instructor_embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    return retriever

yt_link = st.text_input("enter youtube link here!")
def get_text():
    input_text = st.text_input("You: ", "")
    return input_text
if yt_link:
    chain = load_chain(yt_link)
    model = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    user_input = get_text()

if user_input:
    with st.sidebar:
        add_video = st.video(yt_link,start_time=0)
    docs = chain.get_relevant_documents(user_input)
    output = docs[0].page_content
    source_01 = docs[0].metadata.get('source')
    time_sec_01 =docs[0].metadata.get('length')
    answer = model.run(input_documents=docs, question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)
    st.session_state.source.append(source_01)
    st.session_state.time_sec.append(time_sec_01)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        #message(st.session_state["source"][i], key=str(i+99))
        #message(st.session_state["time_sec"][i], key=str(i+999))
        message(st.session_state["past"][i], is_user=True, key=str(i+9999) + "_user")
