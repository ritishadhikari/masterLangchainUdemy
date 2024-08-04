from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
import time
import base64
import streamlit as st

def textDownloader(rawText:str):
    timestr=time.strftime("%Y%m%d-%H%M%S")

    # Encode the raw text in base64 format for file download
    # b64=base64.b64encode(s=rawText.encode()).decode()

    # Create a New File Name with a timestamp
    newFileName="codeReviewAnalysisFile_{}.txt".format(timestr)

    st.markdown(body="Download File 🖼")
    # href=f'<a href="data:file/txt;base64,{b64}" download="{newFileName}">Click Here!!</a>'
    # st.markdown(body=href,unsafe_allow_html=True)
    st.download_button(label="Download the file",
                       data=rawText.encode("utf-8"),
                       file_name=newFileName)