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
import os, pinecone
from apikey import openai_apikey, pinecone_apikey, pinecone_env

#licence setup
os.environ['OPENAI_API_KEY'] = openai_apikey
os.environ['PINECONE_API_KEY'] = pinecone_apikey
os.environ['PINECONE_API_ENV'] = pinecone_env


loader = UnstructuredPDFLoader("D:/web/7/0.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print(f'you have {len(texts)}  documents')
print(f'you have {len(texts[0].page_content)}  characters')

embeddings = OpenAIEmbeddings(openai_api_key=openai_apikey)

pinecone.init(
    api_key=pinecone_apikey,  # find at app.pinecone.io
    environment=pinecone_env  # next to api key in console
)
index_name = "lightnovel"
text_a = ""
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
#docsearch = Pinecone.from_texts([text_a], embeddings, index_name=index_name)

index_list = pinecone.list_indexes()
index = pinecone.Index(index_list[0])
query = "who is defeating whom?"

docs = docsearch.similarity_search(query)

#print(docs[0].page_content[:200])


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=openai_apikey)
chain = load_qa_chain(llm, chain_type="stuff")
query = "who is defeating demon king?"
docs = docsearch.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)

print(answer)
##index = pinecone.Index("lightnovel")
index_list = pinecone.list_indexes()
index = pinecone.Index(index_list[0])
index.describe_index_stats()
##index.delete(deleteAll='true', namespace='lightnovel')
##pinecone.create_index("lightnovel", dimension=1536)
##pinecone.delete_index("lightnovel")








st.title("Custom PDF Search!!!")
prompt = st.text_input("Enter your Topic here!!!")