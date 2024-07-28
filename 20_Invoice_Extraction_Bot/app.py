import streamlit as st
from dotenv import load_dotenv,find_dotenv
import utils

def main():
    load_dotenv(dotenv_path=find_dotenv(filename="../.env"))

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title(body="Invoice Extraction Bot")
    st.subheader(body="I can help you in extracting invoice data")

    # Upload the Invoices (pdf files)
    pdf=st.file_uploader(
        label="Upload Invoices Here. Only PDF files allowed",
        type=["pdf"],
        accept_multiple_files=True
    )
    submit=st.button(label="Extract Data")

    if submit:
        with st.spinner(text="Wait for it"):
            df=utils.createDocs(userPDFList=pdf)
            st.write(df.head())

        st.success(body="Hope I was able to save your time")
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index=False).encode(encoding="utf-8"),
            file_name="invoice_extracted_data.csv",
            mime="text/csv",
            key="download-tools-csv"
        )
        st.success(body="Hope I was able to save your time")
    
    if __name__=="__main__":
        main()

