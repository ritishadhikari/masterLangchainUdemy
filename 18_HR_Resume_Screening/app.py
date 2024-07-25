import streamlit as st
import uuid
import utils,constants

if "unique_id" not in st.session_state:
    st.session_state['unique_id']=""
if "pushToPinecone" not in st.session_state:
    st.session_state["pushToPinecone"]=False

def main():

    st.set_page_config(
        page_title="Resume Screening Assistance..."
    )

    st.title(body="HR - Resume Screening Assistance...")
    st.subheader(body="I can help you in resume screening process")

    jobDescription=st.text_area(label="Please paste the 'Job Description' here...",
                                key="1")
    documentCount=st.number_input(label="Number of resumes to return",key="2")

    # Upload the Resumes (pdf files)
    pdf=st.file_uploader(label="Upload resumes here, only PDF files allowed",
                         type=['pdf'],
                         accept_multiple_files=True
                             )
    
    submit=st.button(label="Help me with the analysis")

    if submit:
        with st.spinner(text="Wait for it..."):
            st.write("Our Process")

            # Creating a unique ID, so that we can use to query and get only the user
            # uploaded documents from PINECONE vector store
            # Generates a string of 32 word length
            if st.session_state['unique_id']=="":
                st.session_state['unique_id']=uuid.uuid4().hex
            # st.write(st.session_state['unique_id'])
            
            # Create a documents list out of all the user uploaded pdf files
            docs=utils.createDocs(userPDFList=pdf,
                             uniqueID=st.session_state['unique_id'])
            st.write(docs)

            # Displaying the count of resumes that have been uploaded
            docCount=f"Total Number of Documents Uploaded: {len(docs)}"
            st.write(docCount)

            # Create embeddings instance
            embeddings=utils.createEmbeddingsLoadData()

            # Push Data to Pinecone
            if st.session_state["pushToPinecone"]==False:
                utils.push_to_pinecone(
                    pineconeAPIKey=constants.PINECONE_API_KEY,
                    embeddings=embeddings,
                    docs=docs,
                    pineconeIndexName=constants.PINECONE_INDEX_NAME
                    )
                st.session_state["pushToPinecone"]=True

           # Fetch relevants documents from PINECONE
            vectorStore=utils.pullFromPineCone(
                pineconeIndexName=constants.PINECONE_INDEX_NAME,
                embeddings=embeddings,
                pineconeAPIKey=constants.PINECONE_API_KEY
            )
            
            similarDocs=vectorStore.similarity_search_with_score(
                query=jobDescription,
                k=documentCount,
                filter={"unique_id":st.session_state['unique_id']}
            )

            # st.write(similarDocs)

            # Introducing a Line Separator
            st.divider()

            # For each Line Item in relevant docs - we are displaying some info in UI
            for item in range(len(similarDocs)):
                st.subheader(body=f"{str(item+1)}")
                
                # Displaying File Path:
                filePath=f"Resume Name: {similarDocs[item][0].metadata['name']}"
                st.write(filePath)
                with st.expander(label="Show me 🎭"):
                    st.info(body=f"Match Score: {similarDocs[item][1]}")
                    summary=utils.getSummary(currentDoc=similarDocs[item][0],
                                             apiKey=constants.GOOGLE_API_KEY)
                    summaryNotice=f"Summary: {summary}"
                    st.write(summaryNotice)
        
        st.success(body="Hope I was able to save your time")

if __name__=="__main__":
    main()