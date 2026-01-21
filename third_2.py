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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

os.environ['LANGCHAIN_PROJECT'] = 'Sequential App2'



pdf_path ='Resume__priya__yadav.pdf'


#load pdf
@traceable(name  = 'load_pdf',tags=['pdf' , 'loader'],metadata={"loader" : "PyPdfLoader"})
def load_pdf(pdf_path):
    loader =PyPDFLoader(pdf_path)
    return loader.load()#one document per page

#2 chunk
@traceable (name = "split_documents")
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    return splitter.split_documents(docs)

# #3 Embeded +index
# emb = OpenAIEmbeddings(model = 'text-embedding-3-small')
@traceable(name='build_vectorstore')
def build_vectorize(splits):
    emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vs = FAISS.from_documents(splits,emb)
    return vs
    # retriever = vs.as_retriever(search_type = 'similiarity',search_kwargs={"k" : 4})

@traceable (name ="setup_pipeline")
def setup_pipeline(pdf_path):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorize(splits)

    return vs

api_key = os.getenv('CHAT_GROQ_KEY')
llm =ChatGroq(model='llama-3.1-8b-instant',api_key=api_key)


#4 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the provided context. If not found, say you do not know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# llm = ChatOpenAI(model= "gpt-4o-mini",temperature = 0 )

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


#build the index under traced setup
vectorstore = setup_pipeline(pdf_path)
retriever = vectorstore.as_retriever(search_type = "similarity",search_kwargs = {"k" : 4})

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




