import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import chromadb
import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from config import (MISTRAL_CHROMA_COLLECTION, OPENAI_CHROMA_COLLECTION,
                    STRANSFORMERS_CHROMA_COLLECTION, EmbeddingType)


class RAG(VectorStore):
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def __init__(self, embedding_type: EmbeddingType) -> None:
        """
        Set up the RAG model with the specified embedding type.
        
        Args:
            embedding_type (EmbeddingType): The type of embedding to use.

        Raises:
            ValueError: If the embedding type is invalid.
        """
        
        match embedding_type:
            case EmbeddingType.OPEN_AI:
                self._langchain_chroma = Chroma(
                    persist_directory=OPENAI_CHROMA_COLLECTION,
                    embedding_function=OpenAIEmbeddings(),
                )
            case EmbeddingType.SENTENCE_TRANSFORMER:
                self._langchain_chroma = Chroma(
                    persist_directory=STRANSFORMERS_CHROMA_COLLECTION,
                    embedding_function=HuggingFaceEmbeddings(),
                )
            case EmbeddingType.MISTRAL:
                self._langchain_chroma = Chroma(
                    persist_directory=MISTRAL_CHROMA_COLLECTION,
                    embedding_function=MistralAIEmbeddings(),
                )
            case _:
                raise ValueError(
                    f"Invalid embedding type: {embedding_type}. Must be one of {EmbeddingType.__members__}"
                )

    def add_texts(
        self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any
    ) -> List[str]:
        return self._langchain_chroma.add_texts(texts, metadatas, **kwargs)

    def embeddings(self) -> Optional[Embeddings]:
        return self._langchain_chroma.embeddings()

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return self._langchain_chroma.delete(ids, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Search for the at most k most similar documents to the query.

        Performs a k-nearest neighbors search on the vector store using the query.
        Filters out any non-similar documents using the cross-encoder.
        Reranks the documents based on the cross-encoder scores.
        
        Args:
            query (str): The query to search for.
            k (int): The maximum number of documents to return. Defaults to 4.

        Returns:
            List[Document]: The at most k most similar documents to the query in order of relevance.
        """

        results = self._langchain_chroma.similarity_search(query, k, **kwargs)
        docs = [doc.page_content for doc in results]
        # print(f"\n query = {query}, docs = {docs}\n") # by EE
        pairs = []
        for doc in docs:
            pairs.append([query, doc])

        scores = RAG._cross_encoder.predict(pairs)

        reordered_docs = []
        for o in np.argsort(scores)[::-1]:
            if scores[0] > 0:
                reordered_docs.append(Document(page_content=results[o].page_content, metadata=results[o].metadata))

        return reordered_docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self._langchain_chroma.similarity_search_by_vector(
            embedding, k, filter, where_document, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self._langchain_chroma.similarity_search_with_score(
            query, k, filter, where_document, **kwargs
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._langchain_chroma._select_relevance_score_fn()

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self._langchain_chroma.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter, where_document, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self._langchain_chroma.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, filter, where_document, **kwargs
        )

    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        return cls._langchain_chroma.from_texts(
            texts,
            embedding,
            metadatas,
            ids,
            collection_name,
            persist_directory,
            client_settings,
            client,
            collection_metadata,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        return cls._langchain_chroma.from_documents(
            documents,
            embedding,
            ids,
            collection_name,
            persist_directory,
            client_settings,
            client,
            collection_metadata,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        return self._langchain_chroma.delete(ids, **kwargs)
