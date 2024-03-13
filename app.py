from enum import IntEnum
from operator import itemgetter
from typing import Dict, Tuple
from dotenv import load_dotenv

from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.documents import Document
from langchain_core.runnables import ConfigurableFieldSpec, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from rag_utils import RAG, EmbeddingType

from prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT

load_dotenv()

VECTOR_STORES = {
    EmbeddingType.OPEN_AI: RAG(embedding_type=EmbeddingType.OPEN_AI),
    EmbeddingType.SENTENCE_TRANSFORMER: Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()),
    EmbeddingType.MISTRAL: Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()),
}

class ChatBot:
    
    store: Dict[Tuple[str, str], BaseChatMessageHistory] = {}

    @staticmethod
    def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in ChatBot.store:
            ChatBot.store[(user_id, conversation_id)] = ChatMessageHistory()
        return ChatBot.store[(user_id, conversation_id)]

    @classmethod
    def format_document(cls, doc: Document) -> str:
        # base_info = {"page_content": doc.page_content, **doc.metadata}
        # document_info = {k: base_info[k] for k in DOCUMENT_PROMPT.input_variables}
        document_info = {
            "page_content": doc.page_content,
            "file_source": "data.pdf, page 5",
        }
        return DOCUMENT_PROMPT.format(**document_info)

    @staticmethod
    def combine_documents(docs, document_separator="\n\n"):
        doc_strings = [ChatBot.format_document(doc) for doc in docs]
        return document_separator.join(doc_strings)


    @classmethod
    def get_vector_store(cls, embedding_type: EmbeddingType) -> VectorStore:
        return VECTOR_STORES[embedding_type]

    @classmethod
    def construct_chain(cls, embedding_type: EmbeddingType, model_type: str):
        vector_store = cls.get_vector_store(embedding_type)
        retriever = vector_store.as_retriever()

        #####################################x
        # INPUTS:                            #
        #   chat_history: ChatMessageHistory #
        #   question: str                    #
        # OUTPUTS:                           #
        #   standalone_question: str         #
        #   chat_history: str                #
        #   question: str                    #
        #####################################x
        standalone_question = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(chat_history=lambda x: get_buffer_string(x["chat_history"]))
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
            question=lambda x: x["question"],
            chat_history=lambda x: x["chat_history"],
        )

        #########################################
        # INPUTS:                               #
        #   standalone_question: str            #
        #   chat_history: str                   #
        #   question: str                       #
        # OUTPUTS:                              #
        #   context: str                        #
        #   standalone_question: str            #
        #   chat_history: str                   #
        #   question: str                       #
        #########################################
        question_context = {
            "context": (
                itemgetter("standalone_question")
                | retriever
                | cls.combine_documents
            ),
            "standalone_question": lambda x: x["standalone_question"],
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }


        ##########################################
        # INPUTS:                                #
        #   chat_history: ChatMessageHistory     #
        #   question: str                        #
        # OUTPUTS:                               #
        #   str (Model answer)                   #
        ##########################################
        rag_chain = (
            standalone_question
            | question_context
            | ANSWER_PROMPT
            | ChatOpenAI()
            | StrOutputParser()
        )

        ################################
        # INPUTS:                      #
        #   question: str              #
        #   configuration: dict        #
        #      user_id: str            #
        #      conversation_id: str    #
        # OUTPUTS:                     #
        #   str                        #
        ################################
        # NOTE: The user_id and conversation_id as a pair defines the session and thus the chat history
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=ChatBot.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

        return with_message_history
    

    chains: Dict[Tuple[EmbeddingType, str], RunnableLambda] = None

    @classmethod
    def get_chain(cls, embedding_type, model_type):
        if not cls.chains:
            cls.chains = {
                (EmbeddingType.OPEN_AI, "openai"): ChatBot.construct_chain(EmbeddingType.OPEN_AI, "openai"),
            } 

        return cls.chains[(embedding_type, model_type)]
        


user_config = {
    "user_id": "user_id",
    "conversation_id": "conversation_id",
}

from langfuse.callback import CallbackHandler

trace = {
    "callbacks": [
        CallbackHandler(
            secret_key="Fill out",
            public_key="Fill out",
            host="http://localhost:3000",
        )
    ]
}

while True:
    # print history
    history = ChatBot.get_session_history(user_config["user_id"], user_config["conversation_id"])
    print(history)

    # get question
    question = input("Enter your question: ")

    # get chain
    chain = ChatBot.get_chain(embedding_type=EmbeddingType.OPEN_AI, model_type="openai")
    response = chain.invoke({"question": question}, config={"configurable": user_config} | trace)


