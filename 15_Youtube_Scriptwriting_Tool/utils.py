from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun

# Function to generate script for Video
def generateScript(prompt,videoLength, creativity, apiKey):

    # Template for generating title
    titleTemplate=PromptTemplate(
        input_variables=['subject'],
        template="Please come up with a title for a YouTube video on the {subject}"
    )

    # Template for Generating "Video Script" using Search engine
    scriptTemplate=PromptTemplate(
        input_variables=["title","DuckDuckGo_Search","duration"],
        template="""
        Create a Script for a Youtube Video based on this title for me.
        Title: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search}
        """,
        input_types={"title":str,"DuckDuckGo_Search":str,"duration":int}
    )

    # Setting up OpenAI LLM
    llm=OpenAI(
               temperature=creativity,
               api_key=apiKey,
               verbose=True)
    
    titleChain=LLMChain(llm=llm, prompt=titleTemplate,verbose=True)
    title=titleChain.invoke(input={"subject":prompt})["text"]
    
    scriptChain=LLMChain(llm=llm, prompt=scriptTemplate,verbose=True)
    search=DuckDuckGoSearchRun()
    searchResult=search.invoke(input=prompt)
    script=scriptChain.invoke(input={"title":title,"duration":videoLength,
                              "DuckDuckGo_Search":searchResult})['text']
    
    return searchResult,title,script