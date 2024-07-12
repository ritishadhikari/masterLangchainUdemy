from dotenv import load_dotenv,find_dotenv
import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

# For efficient Similarity Search and clustering of large scale datasets
# Provides optimized indexing structures and algorithms for tasks like nearest
# neighbours and recommendation system 
from langchain.vectorstores import FAISS

load_dotenv(find_dotenv())


st.set_page_config(
    page_title="Educate Kids",
    page_icon=":robot:"
)

st.header(body="Tell me something and I will give you similar things")

embeddings=OpenAIEmbeddings()

loader=CSVLoader(
    file_path="myData.csv",
    csv_args={
        "delimiter":",",
        "quotechar": '"',
        "fieldnames":['Words']
    }
)

data=loader.load()

print(data)

db=FAISS.from_documents(
    documents=data,
    embedding=embeddings,
)

def getText():
    inputText=st.text_input(label="You: ", key=input)
    return inputText

userInput=getText()
submit=st.button(label="Find Similar Things")

if submit:
    docs=db.similarity_search(query=userInput,k=2)
    st.subheader(body="Top Matches: ")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)