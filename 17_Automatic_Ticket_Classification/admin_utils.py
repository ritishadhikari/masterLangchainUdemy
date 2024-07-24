from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
import pinecone
from langchain_community.vectorstores import Pinecone
import pandas as pd
from sklearn.model_selection import train_test_split

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

#### Functions for dealing with Model Related Tasks ####
def readData(data):
    df=pd.read_csv(filepath_or_buffer=data, 
                   delimiter=",",
                   header=None)
    return df

def createEmbeddings(df:pd.DataFrame,embeddings):
    df[2]=df[0].apply(lambda c: embeddings.embed_query(c))
    return df

def splitTrainTestData(data):
    sentTrain,sentTest, labTrain,labTest=train_test_split(
        list(data[2]), list(data[1]), test_size=0.25, random_state=34
    )
    return sentTrain,sentTest,labTrain,labTest

def getScore(svmClassifier, sentTest, labTest):
    score=svmClassifier.score(sentTest,labTest)
    return score