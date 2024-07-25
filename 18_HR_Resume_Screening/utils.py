from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
import pinecone 
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub
from pypdf import PdfReader

# Extract Information from PDF file
def getPDFText(pdfDoc):
    text=""
    pdfReader=PdfReader(stream=pdfDoc)
    for page in pdfReader.pages:
        text+=page.extract_text()
    return text

# Iterate over files in that user uploaded PDF files, one by one
def createDocs(userPDFList, uniqueID):
    docs=[]
    for filename in userPDFList:
        chunks=getPDFText(pdfDoc=filename)
        docs.append(
            Document(
                page_content=chunks,
                metadata={
                    'name':filename.name,
                    # "id": filename.id,
                    "type": filename.type,
                    "size":filename.size,
                    "unique_id":uniqueID
                    }
            )
        )
    return docs

# Create Embeddings Instance
def createEmbeddingsLoadData():
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# Push Documents to Pinecone
def push_to_pinecone(pineconeIndexName,
                     embeddings,docs,pineconeAPIKey):

    pinecone.Pinecone(api_key=pineconeAPIKey)
    Pinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=pineconeIndexName, 
    )

def pullFromPineCone(pineconeIndexName,
                     embeddings,pineconeAPIKey):
    pinecone.Pinecone(api_key=pineconeAPIKey)
    vectorStore=Pinecone.from_existing_index(
        index_name=pineconeIndexName,
        embedding=embeddings
    )
    return vectorStore

def getSummary(currentDoc,apiKey):
    # llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
    llm=GoogleGenerativeAI(model="gemini-1.5-flash-001",temperature=0.2,
                           google_api_key=apiKey)
    chain=load_summarize_chain(llm=llm,chain_type="map_reduce")
    summary=chain.invoke(input=[currentDoc])
    return summary