import streamlit as st
from utils import (get_website_data,
                   splitData,createEmbeddings,
                   createPineConeIndex,
                   push_to_pinecone,pullFromPineCone,getSimilarDocs)
import constants
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

# Creating Session State Variable
if "HuggingFace_API_Key" not in st.session_state:
    st.session_state["HuggingFace_API_Key"]=""
if "Pinecone_API_Key" not in st.session_state:
    st.session_state['Pinecone_API_Key']=""
if "Pinecone_Index" not in st.session_state:
    st.session_state['Pinecone_Index']=False

st.title(body="🥞 AI Assitance for Website")

### Side Bar Functionality Starts
st.sidebar.title(body="🥙🍣")
st.session_state['HuggingFace_API_Key']=st.sidebar.text_input(
    label="Enter HuggingFace API Key", type="password")
st.session_state["Pinecone_API_Key"]=st.sidebar.text_input(
    label="Enter Pinecone_API_Key",type="password")

loadButton=st.sidebar.button(
    label="Load Data to Pinecone",key="load_button")

if loadButton:
    if (st.session_state['HuggingFace_API_Key'] 
        and st.session_state["Pinecone_API_Key"]):

        siteData=get_website_data(
            sitemap_url=constants.WEBSITE_LINK)
        st.write("Data Pull Done...")

        chunksData=splitData(docs=siteData)
        st.write("Splitting Data Done...")

        embeddings=createEmbeddings()
        st.write("Embeddings instance creation done...")

        if st.session_state['Pinecone_Index'] is False:
            createPineConeIndex(indexName=constants.INDEX_NAME,
                                apiKey=st.session_state['Pinecone_API_Key'])
            st.session_state['Pinecone_Index']=True
            st.write("Index Created Successfully...")
        else:
            st.write("Index has already been created")
            
        
        vectorStore=push_to_pinecone(pineconeIndexName=constants.INDEX_NAME,
                         embeddings=embeddings,
                         docs=chunksData,
                         pineconeEnvironment=constants.PINECONE_ENVIRONMENT,
                         pineconeAPIKey=st.session_state['Pinecone_API_Key'])
        st.write("Pushing Data to Pinecone done...")



        st.sidebar.success("Data pushed to pinecone successfully")
    else:
        st.write("Please Provide the API Keys")


prompt=st.text_input(label="How can I help you?",key="prompt")
documentCount=st.slider(label="Number of Links to return",
                        min_value=0,max_value=5,value=2,step=1)
submit=st.button(label="Search")

if submit:
    if (st.session_state['HuggingFace_API_Key'] 
        and st.session_state["Pinecone_API_Key"]):

        embeddings=createEmbeddings()
        st.write("Embeddings instance creation done...")


        vectorStore=pullFromPineCone(pineconeIndexName=constants.INDEX_NAME,
                                     embeddings=embeddings,
                                     pineconeEnvironment=constants.PINECONE_ENVIRONMENT,
                                     pineconeAPIKey=st.session_state['Pinecone_API_Key'])
        st.write("Pinecone Index Retrieval Done")

        similarDocs=getSimilarDocs(vectorStore=vectorStore,query=prompt,k=documentCount)
        st.success("Please find the Search Results")
        st.write(similarDocs)