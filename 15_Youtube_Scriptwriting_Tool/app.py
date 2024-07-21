import streamlit as st
from utils import generateScript

# Apply Styling
st.markdown(""" 
<style>
            div.stButton > button:first-child {
                background-color: #0099ff;
                color: #ffffff
            }

            div.stbutton > button:hover {
                background-color: #00ff00;
                color: #FFFFFF
            }
</style>
""",unsafe_allow_html=True)

if "API_KEY" not in st.session_state:
    st.session_state["API_KEY"]=""

st.title(body="🤶🏻 Youtube Scriptwriting Tool")

st.sidebar.title(body="👩🏻‍🦱🍖")
st.session_state["API_KEY"]=st.sidebar.text_input(
    label="What's your API Key",
    type="password")
st.sidebar.image(image="./Youtube.jpg",width=300,use_column_width=True)

# Capture User's Inputs
prompt=st.text_input(label="Please provide the topic of the video",key="prompt")
videoLength=st.text_input(
    label="Expected Video Length in Minutes",
    key="video_length",
    )
creativity=st.slider(label="Creativity Limit (0 Low || 1 High)",
                     min_value=0.0,
                     max_value=1.0,
                     step=0.1)

submit=st.button(label="Generate Script")

if submit:
    if st.session_state['API_KEY']:
        searchResult, title, script=generateScript(prompt=prompt,
                                                   videoLength=videoLength,
                                                   creativity=creativity,
                                                   apiKey=st.session_state["API_KEY"])
        st.success(body="Hope You like this script")

        st.subheader(body="Title:🍟")
        st.write(title)

        st.subheader(body="Your Video Script")
        st.write(script)

        st.subheader("Check Out - DuckDuckGo Search: ")
        with st.expander(label="Show me"):
            st.info(body=searchResult)
    else:
        st.error(body="Ooops!!! Please Provide API Key....")