from dotenv import load_dotenv,find_dotenv
import streamlit as st
import user_utils,user_constants

def main():
    load_dotenv(find_dotenv())

    
    st.header(body="Automatic Ticket Classification Tool")
    st.write("We are here to help you, please ask your questions:")
    userInput=st.text_input(label="🎟")

    if userInput:

        # Creating embeddings instance
        embeddings=user_utils.createEmbeddingsLoadData()

        # Function to pull Index Data from Pinecone
        vectorStore=user_utils.pullFromPineCone(
            pineconeAPIKey=user_constants.Pinecone_APIKey,
            embeddings=embeddings,
            pineconeIndexName=user_constants.Pincecone_IndexName
            )

        # Function that will help in fetching the top relevant documents
        # from our vector store - Pinecone Index
        relevantDocs=user_utils.getSimilarDocs(
            vectorStore=vectorStore,
            query=userInput,
            k=3)
        

        # Returns the fine tuned response by LLM
        response=user_utils.getAnswer(
                                    userInput=userInput,
                                    retriever=vectorStore.as_retriever(k=2))
        st.write(response)

if __name__=="__main__":
    main()