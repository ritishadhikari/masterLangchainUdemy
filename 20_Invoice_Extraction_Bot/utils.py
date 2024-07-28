from langchain_google_genai import GoogleGenerativeAI
from pypdf import PdfReader
import pandas as pd
import re
import replicate
from langchain.prompts import PromptTemplate

def getpdfText(pdfDoc):
    text=""
    pdfReader=PdfReader(stream=pdfDoc)
    for page in pdfReader.pages:
        text+=page.extract_text()
    return text

def extractedData(pagesData):
    template="""
    Extract all the following values: invoice no, Description, Quantity, date,
    Unit Price, Amount, Total, Email, Phone Number and Address from this data: {pages}

    Expected Output: Remove any dollar symbols 
    {{ 'Invoice No.': '1001329','Description':'Office Chair', 'Quantity':'2', 'Date':"5/4/2019", "Amount":"2200.00","Email":"ritishadhikari@gmail.com", "phoneNumber":"9434217611","Address":"Kolkata, West Bengal"}}
    """

    promptTemplate=PromptTemplate(
        input_variables=["pages"],
        template=template
    ).format(pages=pagesData)

    # llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0.7)

    # fullResponse=llm.invoke(
    #     input=promptTemplate
    # )
   
    input = {
    "top_p": 0.9,
    "prompt": promptTemplate,
    "temperature": 0.75,
    "max_new_tokens": 800,
    "max_length": 512,
    "repeatation_penalty":1
    }

    fullResponse=""
    for event in replicate.stream(
        ref="meta/llama-2-7b-chat:r8_EPOIbynEWRnVFGOT7PYSBbtyMqnjtQm0pJQuL",
        input=input,
    ):
        fullResponse+=event

    return fullResponse

def createDocs(userPDFList):
    df=pd.DataFrame(
        data={
            "Invoice no.": pd.Series(dtype="str"),
            "Description":pd.Series(dtype="str"),
            "Quantity":pd.Series(dtype="str"),
            "Date":pd.Series(dtype="str"),
            "Unit Price":pd.Series(dtype="str"),
            "Amount":pd.Series(dtype="int"),
            "Total":pd.Series(dtype="int"),
            "Email":pd.Series(dtype="str"),
            "Phone Number":pd.Series(dtype="str"),
            "Address":pd.Series(dtype="str"),
            })
    
    for fileName in userPDFList:
        print(fileName)
        rawData=getpdfText(pdfDoc=fileName),
    

        llmExtractedData=extractedData(pagesData=rawData)
        print(llmExtractedData)
        pattern=r"{(.+)}"
        match=re.search(pattern=pattern,string=llmExtractedData,flags=re.DOTALL)

        if match:
            extractedText=match.group(1)
            dataDict=eval("{" + extractedText + "}")
            print(dataDict)
        else:
            print("No Match Found")

        df=df.append([dataDict],ignore_index=True)
    
    return df