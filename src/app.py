

import shutil
import os

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
tokenizer ,base_model = None, None

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('db'):
    os.makedirs('db')
persist_directory = "./db"
def load_model():
    global tokenizer, base_model
    checkpoint = "./LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map = "cpu",
        torch_dtype = torch.float32
        ) 
# base_model_config =AutoConfig.from_pretrained(
#     checkpoint,
#     device_map = "cpu",
#     torch_dtype = torch.float32
#     )
# base_model = AutoModelForSeq2SeqLM.from_config(base_model_config)

@st.cache_resource
def llm_pipeline():
    global base_model,tokenizer
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length =1024, # 256/1024
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline = pipe)
    return local_llm

@st.cache_resource 
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory = "./db", embedding_function = embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = retriever,
        return_source_documents = True
    )
    return qa

def process_answer(instruction):
    print("Processing answer")
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def streamlit_main():

    st.title('Local Lease 📄🦜')
    with st.expander('About the app'):
        st.markdown(
            """
            This is generative ai powered personal legal document assistant app that responds to questions about apartment leasing agreements.
    """
        )
    
    question = st.text_area('Ask a question about your lease')
    with st.status("loading the model..."):
        load_model()
    if st.button('Search'):
        st.info("Your question: "+question)
        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

def generate_vectors_from_pdf(): 
    for root,dirs,files in os.walk("./data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root,file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)
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
def functional_main(): 
    # take pdf filepath from cli
    pdf_save_folder = "./data"
    pdf_path = input("Enter the path to the pdf file: ")
    shutil.copy(pdf_path, pdf_save_folder)
    print("File uploaded successfully")
    print("==============================================================")
    print("Generating vectors from pdf...")
    generate_vectors_from_pdf()
    print("Vectors generated successfully")
    print("==============================================================")
    print("Loading model...")
    load_model()
    print("Model loaded successfully")
    print("==============================================================")
    while True:
        question = input("Ask a question about your lease: ")
        answer, metadata = process_answer(question)
        print("\n\nYour Answer")
        print(answer)
        print("\n\nMetadata")   
        print(metadata)
        print("==============================================================")

    

if __name__ == "__main__":
    # streamlit_main()
    functional_main()