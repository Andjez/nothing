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

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

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
    store = Chroma.from_texts(texts, instructor_embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    #RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2, ),chain_type="stuff",retriever=retriever,return_source_documents=True)
    #chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    return retriever#chain

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
    output = chain.get_relevant_documents(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
