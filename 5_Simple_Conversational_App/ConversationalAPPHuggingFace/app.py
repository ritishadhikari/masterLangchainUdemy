import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage, 
    HumanMessage,
    SystemMessage
)

st.set_page_config(
    page_title="Langchain Demo",
    page_icon=":robot:"
)

chat=ChatOpenAI(temperature=0)

st.header("Hey I am your ChatGPT")

if "sessionMessages" not in st.session_state:
    st.session_state['sessionMessages']=[
        SystemMessage(content=
                     """
                         You are a Helpful Assistant
                     """
                     )
        ]

def loadAnswer(question):
    st.session_state["sessionMessages"].append(
        HumanMessage(content=question)
    )
    assistantAnswer=chat.invoke(input=st.session_state['sessionMessages'])
    st.session_state['sessionMessages'].append(AIMessage(content=assistantAnswer.content))
    return assistantAnswer.content

def getText():
    inputText=st.text_input(label="You: ")
    return inputText

userInput=getText()
submit=st.button(label="Generate")

if submit:
    response=loadAnswer(question=userInput)
    st.subheader("Response: ")
    st.write(response)
