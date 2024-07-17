import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv,find_dotenv
from langchain_openai import OpenAI,ChatOpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              ConversationSummaryMemory,
                              ConversationTokenBufferMemory
                              )
import tiktoken
from langchain.chains import ConversationChain

load_dotenv(dotenv_path=find_dotenv(filename="../../.env"))

if "conversation" not in st.session_state:
    st.session_state['conversation']=None
if 'messages' not in st.session_state:
    st.session_state['messages']=[]
if "apiKey" not in st.session_state:
    st.session_state['apiKey']=""

# Setting Page title and header
st.set_page_config(
    page_title="Chat GPT Clone",
    page_icon=":robot_face:"
)

st.markdown(
    """
    <h1 style='text-align: center>  
        How can I assist you?
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("👱🏼‍♂️")
st.session_state['apiKey']=st.sidebar.text_input("What's Your API Key?", type="password")

summarize_button=st.sidebar.button(
    "Summarize the conversation",
    key="summarise"
)

if summarize_button:
    st.sidebar.write("Nice Chatting with you my friend :\n\n"+"Hello Friend")

def getResponse(userInput,apiKey):
    if st.session_state['conversation'] is None:
        llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo",api_key=apiKey)

        st.session_state['conversation']=ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory()
        )

    response=st.session_state['conversation'].invoke(input=userInput)
    print(st.session_state['conversation'].memory.buffer)
    return response
    

responseContainer=st.container()
container=st.container()

with container:
    with st.form(
        key="my_form",
        clear_on_submit=True):
        userInput=st.text_area(
            "Your Question Goes Here:",
            key="input",
            height=100
        )
        submitButton=st.form_submit_button(label="Send")
        if submitButton:
            st.session_state['messages'].append(userInput)
            # st.write(st.session_state['messages'])
            answer=getResponse(
                userInput=userInput,
                apiKey=st.session_state['apiKey'])['response']
            st.session_state['messages'].append(answer)
            with responseContainer:
                # st.write(answer)
                for i in range(len(st.session_state['messages'])):
                    if i%2==0:
                        message(
                            message=st.session_state['messages'][i],
                            is_user=True,
                            key=str(i)+"_user"
                            )
                    else:
                        message(
                            message=st.session_state['messages'][i],
                            is_user=False,
                            key=str(i)+"_AI"
                            )