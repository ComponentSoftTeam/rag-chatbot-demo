from abc import abstractmethod
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from dataclasses import dataclass
from enum import Enum
import os
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from typing import List

import chromadb

class EmbeddingType(Enum):
    OPEN_AI = "openai"
    SENTENCE_TRANSFORMER = "sentence transformers"
    MISTRAL = "mistral"
    
@dataclass
class RagResponse:
    source: str
    content: str

class RAG:
    @staticmethod
    def create(embedding_type: EmbeddingType = EmbeddingType.OPEN_AI):
        match(embedding_type):
            case EmbeddingType.OPEN_AI:
                return RAG_OPENAI()
            case _:
                raise NotImplementedError(f"Embedding type {embedding_type} not implemented")
            
    @abstractmethod
    def exec_query(
        self,
        prompt: str,
        max_files: int = 1,
        ) -> List[RagResponse]:
        """Given a prompt, retuns the relevant file contents

        Args:
            prompt (str): _description_
            max_files (int): _description_

        Returns:
            List[RagResponse]: Sorted list based on relevance
        """


    
    
class RAG_OPENAI(RAG):
    def __init__(self) -> None:
        self._langchain_chroma = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    

    def exec_query(self, prompt: str, max_files: int = 1) -> List[RagResponse]:
        returned_docs = self._langchain_chroma.similarity_search(prompt, max_files)

        rag_responses = [RagResponse(content=doc.page_content, source="") for doc in returned_docs]
        
        return rag_responses
    

load_dotenv(find_dotenv())
print(os.environ.get("OPENAI_CHROMA_COLLECTION") if os.environ.get("OPENAI_CHROMA_COLLECTION") else "No collection")
rag = RAG.create(EmbeddingType.OPEN_AI)

print(f"There are {rag._langchain_chroma._collection.count()} records in the collection {os.environ.get('OPENAI_CHROMA_COLLECTION')}")

print(rag.exec_query("What frequency bands are used in 5G?", 3))