from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone 
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone
import asyncio
from langchain_community.document_loaders import SitemapLoader

def get_website_data(sitemap_url):
    loop=asyncio.new_event_loop()
    asyncio.set_event_loop(loop=loop)
    # Sitemaploader loads a sitemap from a given URL and then scrape and load
    # all pages in the sitemap, returning each page as a document
    loader=SitemapLoader(web_path=sitemap_url)
    docs=loader.load()
    return docs

def splitData(docs):
    textSplitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docsChunks=textSplitter.split_documents(documents=docs)
    return docsChunks

def createEmbeddings():
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def createPineConeIndex(indexName,apiKey):
    """
    Only run for the first time
    """
    pc=pinecone.Pinecone(api_key=apiKey)
    if indexName not in pc.list_indexes().names():
        pc.create_index(
            name=indexName,
            dimension=384,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

def push_to_pinecone(pineconeIndexName,
                     embeddings,docs,pineconeAPIKey,pineconeEnvironment):

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

def getSimilarDocs(vectorStore,query,k=2):
    similarDocs=vectorStore.similarity_search(query,k=k)
    return similarDocs