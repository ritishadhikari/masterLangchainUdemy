from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
import pinecone
from langchain.vectorstores import Pinecone

def readPDFData(pdf_file):
    pdf_page=PdfReader(stream=pdf_file)
    text=""
    for page in pdf_page.pages:
        text+=page.extract_text()
    return text

def splitData(text):
    textSplitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=50)
    docs=textSplitter.split_text(text=text)
    docs_chunks=textSplitter.create_documents(texts=docs)
    return docs_chunks

def createEmbeddingsLoadData():
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings

def createPineConeIndex(indexName,apiKey,indexDimension):
    """
    Only run for the first time
    """
    pc=pinecone.Pinecone(api_key=apiKey)
    if indexName not in pc.list_indexes().names():
        pc.create_index(
            name=indexName,
            dimension=indexDimension,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

def push_to_pinecone(pineconeIndexName,
                     embeddings,docs,pineconeAPIKey):

    pinecone.Pinecone(api_key=pineconeAPIKey)
    vectorStore=Pinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=pineconeIndexName, 
    )
    return vectorStore


def pullFromPineCone(pineconeIndexName,
                     embeddings,pineconeAPIKey,pineconeEnvironment):
    pinecone.Pinecone(api_key=pineconeAPIKey)
    vectorStore=Pinecone.from_existing_index(
        index_name=pineconeIndexName,
        embedding=embeddings
    )
    return vectorStore