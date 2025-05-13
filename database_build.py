from langchain_community.document_loaders import TextLoader,JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
import json
import os


os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ['CURL_CA_BUNDLE'] = ''

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["action"] = record.get("action")
    metadata["step"] = record.get("step")
    return metadata


class DocumentSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None
        self.store = LocalFileStore('./vector_cache')
        self.is_load = False
        self.loader = None

    def load_documents(self, database_path):
        """Load documents from a list of dictionaries and build the embedding database.
        
        Args:
            data_list (list): List of dictionaries with 'state' and 'action' keys.
        """
        self.database_path = database_path
        self.loader = JSONLoader(
            file_path=database_path,
            jq_schema=".[]",
            content_key="state",
            metadata_func=metadata_func,
        )
        documents = self.loader.load()
        self.is_load = True
        self.db = Chroma.from_documents(documents, self.embedding)

    def search_by_query(self, query,k=1):
        if self.db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        docs = self.db.similarity_search(query,k)
        return docs
    
    def search_by_query_scores(self, query,k=1):
        if self.db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        docs = self.db.similarity_search_with_relevance_scores(query, k)
        return docs
    
    def search_by_query_batch(self, query_list,k=1):
        """Batch search by multiple queries."""
        if self.db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        results = {}
        for query in query_list:
            docs = self.db.similarity_search(query, k)
            results[query] = docs
        return results
    
    def search_by_vector(self, embedding_vector,k=1):
        if self.db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        docs = self.db.similarity_search_by_vector(embedding_vector,k)
        return docs
    
    def search_by_vector_batch(self, embedding_vector_list,k=1):
        if self.db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")

        results = {}
        for embedding_vector in embedding_vector_list:
            docs = self.db.similarity_search_by_vector(embedding_vector,k)
            results[embedding_vector] = docs
        return results