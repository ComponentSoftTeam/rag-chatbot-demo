from operator import itemgetter

from typing import Iterator
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from operator import itemgetter
from typing import List

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


from abc import abstractmethod

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class ChatBot:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""\
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """
    )

    ANSWER_PROMPT = ChatPromptTemplate.from_template("""\
    Answer the question based only on the following context from files:
    {context}

    Question: {question}
    """
    )

    DOCUMENT_PROMPT = PromptTemplate.from_template("Source {file_source}: {page_content}")


    # def _combine_documents(
    #     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    # ):
    #     doc_strings = [format_document(doc, document_prompt) for doc in docs]
    #     return document_separator.join(doc_strings)


    @classmethod
    def exec_stream(
        cls,
        # embedding_type: EmbeddingType,
        # condensation_chat_model: ChatOpenAI,
        # chat_model: ChatOpenAI,
        # chat_history: List[Tuple[str, str]],
        session_id: str,
        prompt: str,
    ) -> Iterator[str]:

        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

        retriever = vector_store.as_retriever()

        inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(chat_history=lambda x: get_buffer_string(x["chat_history"]))
            | cls.CONDENSE_QUESTION_PROMPT
            | ChatOpenAI()
            | StrOutputParser(),
        )

        return inputs 
        _context = {
            "context": itemgetter("standalone_question") | retriever | cls._combine_documents,
            "question": lambda x: x["standalone_question"],
        }

        conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

        conversational_qa_chain.invoke("aésdlkfj")



res = ChatBot.exec_stream("1", "hey")
res.invoke({"chat_history": "hey", "question": "hey"})


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]



inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(chat_history=lambda x: get_buffer_string(x["chat_history"]))
    | cls.CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)