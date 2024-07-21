from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone 
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader

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