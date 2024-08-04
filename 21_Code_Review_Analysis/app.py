from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
from io import StringIO
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
import utils

load_dotenv(find_dotenv("../.env"))

st.title(body="Let's do Code Review for your python code")
st.header(body="Please Upload your .py file here:")
data=st.file_uploader(label="Upload Python File",type=".py")

if data:
    fetchedData=StringIO(initial_value=data.getvalue().decode(encoding="utf-8")).read()
    st.write(str(fetchedData))

    chat=GoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.9,
                max_tokens=None,
                timeout=None,
                max_retries=2)

    # chat=ChatOpenAI(model="gpt-3.5-turbo",
    #                 temperature=0.9)

    systemMessage=SystemMessage(
        content="""
        You are a code review assistant. Provide detailed suggestions to improve the 
        given Python Code
        """
        )
    
    humanMessage=HumanMessage(content=fetchedData)

    prompt=ChatPromptTemplate.from_messages(
        messages=[systemMessage,humanMessage]
    )

    finalResponse=chat.invoke(input=[systemMessage,humanMessage])

    st.markdown(finalResponse)
    utils.textDownloader(rawText=finalResponse)