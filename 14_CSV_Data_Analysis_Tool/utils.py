from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain_openai import OpenAI

def queryAgents(data, query):

    # Parse the CSV file and create a pandas dataframe from its contents
    df=pd.read_csv(filepath_or_buffer=data)
    llm=OpenAI(model="gpt-3.5-turbo-instruct")
    agent=create_pandas_dataframe_agent(llm=llm,df=df,verbose=True)

    return agent.invoke(input=query)

