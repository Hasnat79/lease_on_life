import os


from langchain_community.llms import Ollama
from openai import OpenAI
from huggingface_hub import hf_hub_download

from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyPDFium2Loader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from utils.llms import Mixtral_7B, Llama2_chat_7B
# HUGGINGFACEHUB_API_TOKEN = getpass("Enter your Hugging Face API token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
def main(): 

    pdf_path = "/scratch/user/hasnat.md.abdullah/689/lease_on_life/src/data/Lease.pdf"

    loader = PyPDFium2Loader(pdf_path)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    #embeddings = OllamaEmbeddings(model="mistral:7b")
    embeddings = OpenAIEmbeddings()
    mixtral_7b = Mixtral_7B(model_kwargs={'temparature':0.6,'max_length':100}).llm

    docsearch = Chroma.from_documents(texts, embeddings)
    print("ok")

# # running hugging face models using lLM Chain and template
#     template = '''Question: {question}
#     Answer: Let's think step by step
#     '''

#     mixtral_7b = Mixtral_7B(model_kwargs={'temparature':0.6,'max_length':100}).llm
#     prompt = PromptTemplate(template = template, input_variables = {'question'})
#     # print(prompt)
#     llm_chain = LLMChain(llm = mixtral_7b, prompt = prompt)
#     print(llm_chain.invoke(question))

if __name__ == '__main__':
    main()