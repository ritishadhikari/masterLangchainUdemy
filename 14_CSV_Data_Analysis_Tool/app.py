import streamlit as st
from dotenv import load_dotenv,find_dotenv
from utils import queryAgents

load_dotenv(find_dotenv(filename="../.env"))

st.title(body="Let's do some analysis on your CSV")
st.header(body="Please Upload your CSV File here:")

# Capture the CSV File
data=st.file_uploader(label="Upload CSV File",type="csv")

query=st.text_area(label="Enter You Query")
button=st.button(label="Generate Response")

if button:
    # Get Response
    answer=queryAgents(data=data,query=query)
    st.write(answer)