import streamlit as st
from utils import get_website_data,splitData,createEmbeddings

# Creating Session State Variable
if "HuggingFace_API_Key" not in st.session_state:
    st.session_state["HuggingFace_API_Key"]=""
if "Pinecone_API_Key" not in st.session_state:
    st.session_state['Pinecone_API_Key']=""

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
            sitemap_url="https://jobs.excelcult.com/wp-sitemap-posts-1.xml")
        st.write("Data Pull Done...")

        chunksData=splitData(docs=siteData)
        st.write("Splitting Data Done...")

        embeddings=createEmbeddings()
        st.write("Embeddings instance creation done...")

        st.write("Pushing Data to Pinecone done...")

        st.sidebar.success("Data pushed to pinecone successfully")
    else:
        st.write("Please Provide the API Keys")
