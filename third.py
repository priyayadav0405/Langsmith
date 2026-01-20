from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
# from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


os.environ['LANGCHAIN_PROJECT'] = 'Sequential App'



pdf_path ='Resume__priya__yadav.pdf'
#load pdf
loader =PyPDFLoader(pdf_path)
docs = loader.load()#one document per page

#2 chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
splits=splitter.split_documents(docs)

# #3 Embeded +index
# emb = OpenAIEmbeddings(model = 'text-embedding-3-small')
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vs = FAISS.from_documents(splits,emb)
retriever = vs.as_retriever(search_type = 'similarity',search_kwargs={"k" : 4})

#4 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the provided context. If not found, say you do not know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# llm = ChatOpenAI(model= "gpt-4o-mini",temperature = 0 )
api_key = os.getenv('CHAT_GROQ_KEY')
llm =ChatGroq(model='llama-3.1-8b-instant',api_key=api_key)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough()
})

chain  = parallel | prompt | llm | StrOutputParser()

# 6 Ask questions
print("PDF RAG ready. Ask a question for ctrl+c to exit ")
q = input("\n Q : ")
ans = chain.invoke(q.strip())

print("\nA",ans)




