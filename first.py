from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
load_dotenv()

#simple one line prompt
prompt = PromptTemplate.from_template("{question}")

api_key = os.getenv('CHAT_GROQ_KEY')
model =ChatGroq(model='llama-3.1-8b-instant',api_key=api_key)
parser = StrOutputParser()

#chain : prompt->model->parser
chain = prompt| model| parser

#run it 
result = chain.invoke({"question" : "what is the capital of india?"})

print(result)