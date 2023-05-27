import streamlit as st

# Set a title for the web app
st.title("Simple Streamlit App")

# Add a text input box
user_input = st.text_input("Enter your name")

"""This is the logic for ingesting Notion data into LangChain."""
import pickle
import faiss
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
#from apikey import openai_apikey

#licence setup
#os.environ['OPENAI_API_KEY'] = openai_apikey

# Here we load in the data in the format that csv exports it in.
#ps = list(Path("raw_data/INDIA/").glob("**/*.csv"))

#data = []
#sources = []
#for p in ps:
  #  with open(p) as f:
 #       data.append(f.read())
#    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
#text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
#docs = []
#metadatas = []
from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",model_kwargs={"device": "cuda"})

#for i, d in enumerate(data):
#    splits = text_splitter.split_text(d)
#    docs.extend(splits)
#    metadatas.extend([{"source": sources[i]}] * len(splits))

docs = "metadatas.extend([{"source": sources[i]}] * len(splits))"
# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, instructor_embeddings)#, metadatas=metadatas)
faiss.write_index(store.index, "in_docs.index")
store.index = None
with open("in_faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

