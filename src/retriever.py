import src.config 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

urls = [
    "https://www.ibm.com/think/topics/large-language-models",
    "https://www.ibm.com/think/topics/retrieval-augmented-generation",
    "https://www.ibm.com/think/topics/neural-networks",
    "https://www.ibm.com/think/topics/transfer-learning",
]

print("Loading documents into the knowledge base...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, 
    chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2"),
)

retriever = vectorstore.as_retriever()