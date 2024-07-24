import streamlit as st
from dotenv import load_dotenv,find_dotenv
from pages import admin_constants
from pages import admin_utils

def main():
    load_dotenv()

    if "pineconeIndexCreation" not in st.session_state:
        st.session_state['pineconeIndexCreation']=False

    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title(body="Please upload your files...")

    # Upload the PDF File
    pdf=st.file_uploader(label="Only PDF files allowed",
                         type=['pdf'])
    
    # Extract the whole text from the uploaded PDF file
    if pdf is not None:
        with st.spinner(text="Loading File"):

            text=admin_utils.readPDFData(pdf_file=pdf)
            st.write("🥨 Reading PDF Done")

            # Create Chunks
            docsChunks=admin_utils.splitData(text=text)
            st.write("🍜 Splitting Data into Chunks Done")

            # Create the Embeddings
            embeddings=admin_utils.createEmbeddingsLoadData()
            st.write("🍚 Creating Embedding instance done")

            # Creating the Index if Index Not yet created
            if st.session_state['pineconeIndexCreation']==False:
                st.session_state['pineconeIndexCreation']=True
                admin_utils.createPineConeIndex(
                    indexName=admin_constants.Pincecone_IndexName,
                    apiKey=admin_constants.Pinecone_APIKey,
                    indexDimension=admin_constants.PineCone_Index_Dimension
                    )
                st.write("Created the Index")
                
            else:
                st.write("Index Already Created; Bypassing Index Creation")

            # Build the Vector Store (Push the PDF data embeddings)
            admin_utils.push_to_pinecone(
                pineconeIndexName=admin_constants.Pincecone_IndexName,
                embeddings=embeddings,
                docs=docsChunks,
                pineconeAPIKey=admin_constants.Pinecone_APIKey 
            )
            st.success(body="Successfully pushed the embeddings to Pinecone")

if __name__=="__main__":
    main()