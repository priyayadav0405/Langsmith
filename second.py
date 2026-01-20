from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential App'

api_key = os.getenv('CHAT_GROQ_KEY')
model1 =ChatGroq(model='llama-3.1-8b-instant',api_key=api_key)
model2 =ChatGroq(model='llama-3.1-8b-instant',api_key=api_key,temperature=0.6)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2  = PromptTemplate(
    template='Genrerate a 5 pointer summary from the following text \n {text}',
    input_variables=['topic']
)



parser = StrOutputParser()

chain = prompt1 | model1 |parser | prompt2 | model2 |parser


config = {
    'run_name':'sequential chain',
    'tags' : ['llm app', 'report generation' , 'summarization'],
    'metadata':{'model1': 'llama-3.1-8b-instant' ,'model1_temp':0.7,'parser':'stroutputparser'}
}

result  = chain.invoke({'topic' : 'Unemployement in India'} ,config=config)

print(result)