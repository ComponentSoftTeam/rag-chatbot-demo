
from dataclasses import dataclass
from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings


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


    @classmethod
    def generate(cls,
                 embedding_model: Embeddings,
                 condensation_chat_model: ChatOpenAI,
                 chat_model: ChatOpenAI,
                 vector_store: Chroma):
        return cls()

    def get_condenstion_chat_model(self):
        return ChatOpenAI(temperature=0)
    
    def get_retriever(self):
        return self.vector_store.as_retriever()
    
    def get_chat_model(self):
        return ChatOpenAI()
    
    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)


    def __init__(self):
        self.vector_store = get_vector_store(OPEAI)  # type: ignore

        retriever = self.get_retriever()

        # creat a vector store from documents
        retriever = VectorStore.from_documents(
            documents=documents, vector_store=vector_store
        )

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(chat_history=lambda x: get_buffer_string(x["chat_history"]))
            | cls.CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        )

        _context = {
            "context": itemgetter("standalone_question") | retriever | self._combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

        return conversational_qa_chain




