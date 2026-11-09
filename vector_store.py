from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
        Creates and returns a FAISS vector store from document chunks.
    """

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    vector_store=FAISS.from_documents(chunks,embeddings)

    return vector_store