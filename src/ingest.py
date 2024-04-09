from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTING
# import chromadb
from chromadb.config import Settings
persist_directory = "./db"
def main(): 
    for root,dirs,files in os.walk("./data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root,file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    # db = chromadb.PersistentClient(
    #     path = persist_directory,
    #     settings=Settings(allow_reset = True)
    # )
    db.persist()
    db = None

   
    

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()
