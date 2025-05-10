import os
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

def get_vector_db():
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL, show_progress=True)
    return Chroma(
            collection_name=COLLECTION_NAME, 
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding
           )
