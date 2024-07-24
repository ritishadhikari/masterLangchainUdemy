import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain_google_genai import GoogleGenerativeAI
import joblib


def pullFromPineCone(pineconeIndexName,
                     embeddings,pineconeAPIKey):
    pinecone.Pinecone(api_key=pineconeAPIKey)
    vectorStore=Pinecone.from_existing_index(
        index_name=pineconeIndexName,
        embedding=embeddings
    )
    return vectorStore

def createEmbeddingsLoadData():
    embeddings=SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-V2")
    return embeddings

def getSimilarDocs(vectorStore,query,k=2):
    similarDocs=vectorStore.similarity_search(query,k=k)
    return similarDocs

def getAnswer(userInput,retriever):
    # chain=load_qa_chain(
    #     llm=GoogleGenerativeAI(model="gemini-pro"),
    #     # llm=OpenAI(model="gpt-3.5-turbo-instruct"),
    #     # chain_type="stuff",
    #     retriever=retriever

    # )

    chain=RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="gemini-pro"),
        chain_type="stuff",
        retriever=retriever
    )

    response=chain.invoke(
                {'query':userInput})['result']
    return response


def predict(queryResult):
    fitModel=joblib.load(filename="modelSVM.pkl")
    result=fitModel.predict([queryResult])[0]
    return result