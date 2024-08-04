import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def getLLMResponse(formInput,emailSender,emailRecipient,emailStyle,
                   startDate,endDate,backup):
    llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest")

    template="""
    Write a email with {style} style and includes topic :{email_topic}.\n\nSender: {sender}\nRecipient: {recipient}
    \nStart Date:{startDate} \nEnd Date:{endDate} \nBackUp Person's Name: {backup}
    Don't mention Backup person's phone number and mail address in the email

    \n\nEmail Text:
    """

    prompt=PromptTemplate(
        input_variables=['style','email_topic','sender','recipient',"backup"],
        template=template
    ).format(style=emailStyle,email_topic=formInput,
             sender=emailSender,recipient=emailRecipient,
             startDate=startDate,endDate=endDate,backup=backup)

    response=llm.stream(input=prompt)
    
    return response

st.set_page_config(
    page_title="Generate Emails",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header(body="Generate Emails 📧")

formInput=st.text_area(label="Enter the Email Topic", height=275)


col1, col2, col3=st.columns(spec=[10,10,5])
with col1:
    emailSender=st.text_input(label="Sender Name")
    emailRecipient=st.text_input(label="Recipient Name")
with col2:
    startDate=st.date_input(label="Enter Leave Start Date")
    endDate=st.date_input("Enter Leave End Date")
with col3:
    backupPerson=st.text_input(label="Backup Person's Name")
    emailStyle=st.selectbox(label="Writing Style",
                            options=["Formal","Appreciating","Not Satisfied","Neutral"])
    
submit=st.button(label="Generate")

if submit:
    st.divider()
    st.write_stream(stream=getLLMResponse(
        formInput=formInput,
        emailSender=emailSender,
        emailRecipient=emailRecipient,
        emailStyle=emailStyle,
        startDate=startDate,
        endDate=endDate,
        backup=backupPerson))
    