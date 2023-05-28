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
import faiss
import pickle
import os, re
#import argparse
import streamlit as st
from langchain import OpenAI
from apikey import openai_apikey
from streamlit_chat import message
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
os.environ['OPENAI_API_KEY'] = openai_apikey
#embedding
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})

def extract_video_code(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:embed\/|watch\?v=)|youtu\.be\/)([\w-]+)"
    match = re.match(pattern, url)

    if match:
        return match.group(1)
    else:
        return None

# Setup

yt_link = st.text_input("enter youtube link here!")
if yt_link:
    yt_id = extract_video_code(yt_link)
    transcript = YouTubeTranscriptApi.get_transcript(yt_id)
    my_dict = str(transcript)
    data = []
    source = []

    # Extract 'text' and 'start' values from each dictionary item
    for item in transcript:
        data.append(item['text'])
        source.append(item['start'])

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": source[i]}] * len(splits))

    store = FAISS.from_texts(data, instructor_embeddings, metadatas=metadatas)
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)



##    parser = argparse.ArgumentParser(description='Ask a question to the youtube video.')
##    parser.add_argument('question', type=str, help='The question to ask the youtube video')
##    args = parser.parse_args()
##
##    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever())
##
##    result = chain({"question": args.question})
##
##    st.write(f"Answer: {result['answer']}")
##    st.write(f"Sources: {result['sources']}")

# Load the LangChain.
@st.cache_resource
def load_chain():
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=store.as_retriever())
    return chain


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "")
    return input_text

user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


