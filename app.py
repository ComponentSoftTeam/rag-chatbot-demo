import uuid
from operator import itemgetter
from typing import Dict, Literal, Tuple, Union, get_args, overload
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
from langfuse.callback import CallbackHandler

from rag_utils import RAG, EmbeddingType

from prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT

load_dotenv()


from langchain_community.llms import Replicate
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI


class ChatBotConfig:
    MODEL_FAMILY = Literal["GPT", "Mistral", "Llama"]

    OPENAI_MODELS = Literal["gpt-3.5-turbo", "gpt-4"]
    MISTRAL_MODELS = Literal["mistral-tiny", "mistral-small", "mistral-medium",  "mistral-large"]
    __LLAMA_MODEL_VERSIONS = {
        "llama-2-7b-chat": "f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4",
        "llama-2-13b-chat": "6b4da803a2382c08868c5af10a523892f38e2de1aafb2ee55b020d9efef2fdb8",
        "llama-2-70b-chat": "2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
    }
    
    LLAMA_MODELS = Literal["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"]
    MODELS = Union[OPENAI_MODELS, MISTRAL_MODELS, LLAMA_MODELS]

    def get_model_name(self, model_family: MODEL_FAMILY, model: MODELS) -> str:
        match model_family:
            case "GPT":
                return model
            case "Mistral":
                match model:
                    case "mistral-tiny":
                        return "mistral-tiny"
                    case "mistral-small":
                        return "mistral-small-latest"
                    case "mistral-medium":
                        return "mistral-medium-latest"
                    case "mistral-large":
                        return "mistral-large-latest"
                    case _: 
                        raise ValueError(f"Invalid model: {model}. Must be one of {get_args(ChatBotConfig.MISTRAL_MODELS)}")
            
            case "Llama":
                # TODO(Kristofy): change model to the per token billed one
                # return f"meta/{model}"
                return f"meta/{model}:{ChatBotConfig.__LLAMA_MODEL_VERSIONS[model]}"
            case _: 
                raise ValueError(f"Invalid model family: {model_family}. Must be one of {get_args(ChatBotConfig.MODEL_FAMILY)}")

    @overload
    def __init__(cls, model_family: Literal["GPT"], model: OPENAI_MODELS): ...

    @overload
    def __init__(cls, model_family: Literal["Mistral"], model: MISTRAL_MODELS): ...

    @overload
    def __init__(cls, model_family: Literal["Llama"], model: LLAMA_MODELS): ...

    def __init__(self, model_family: MODEL_FAMILY, model: MODELS):
        self.model_family = model_family
        self.model = self.get_model_name(model_family, model)

        match model_family:
            case "GPT":
                self.embedding_type = EmbeddingType.OPEN_AI
                if model not in get_args(ChatBotConfig.OPENAI_MODELS):
                    raise ValueError(f"Invalid model: {model}. Must be one of {get_args(ChatBotConfig.OPENAI_MODELS)}")
            case "Mistral":
                self.embedding_type = EmbeddingType.MISTRAL
                if model not in get_args(ChatBotConfig.MISTRAL_MODELS):
                    raise ValueError(f"Invalid model: {model}. Must be one of {get_args(ChatBotConfig.MISTRAL_MODELS)}")
            case "Llama":
                self.embedding_type = EmbeddingType.SENTENCE_TRANSFORMER
                if model not in get_args(ChatBotConfig.LLAMA_MODELS):
                    raise ValueError(f"Invalid model: {model}. Must be one of {get_args(ChatBotConfig.LLAMA_MODELS)}")
            case _: 
                raise ValueError(f"Invalid model family: {model_family}. Must be one of {get_args(ChatBotConfig.MODEL_FAMILY)}")

    def get_condensation_model(self) -> Union[ChatOpenAI, ChatMistralAI, Replicate]:
        match self.model_family:
            case "GPT":
                return ChatOpenAI(
                    model=self.model,
                    temperature=0,
                )

            case "Mistral":
                return ChatMistralAI(
                    model=self.model,
                    temperature=0,
                )

            case "Llama":
                # TODO:(Kristóf) Add version_obj to the Replicate model to avoid the id field
                return Replicate(
                    model=self.model,
                    model_kwargs={"temperature": 0.01},
                    # version_obj=""
                )
                
            case _: 
                raise ValueError(f"Invalid model family: {self.model_family}. Must be one of {get_args(ChatBotConfig.MODEL_FAMILY)}")

    def get_chat_model(self):
        match self.model_family:
            case "GPT":
                return ChatOpenAI(
                    model=self.model,
                    temperature=0.7,
                )

            case "Mistral":
                return ChatMistralAI(
                    model=self.model,
                    temperature=0.7,
                )

            case "Llama":
                # TODO:(Kristóf) Add version_obj to the Replicate model to avoid the id field
                return Replicate(
                    model=self.model,
                    model_kwargs={"temperature": 0.7},
                    # version_obj=""
                )
                
            case _: 
                raise ValueError(f"Invalid model family: {self.model_family}. Must be one of {get_args(ChatBotConfig.MODEL_FAMILY)}")

        
class ChatBot:
    
    VECTOR_STORES = {
        EmbeddingType.OPEN_AI: RAG(embedding_type=EmbeddingType.OPEN_AI),
        EmbeddingType.SENTENCE_TRANSFORMER: RAG(embedding_type=EmbeddingType.SENTENCE_TRANSFORMER),
        EmbeddingType.MISTRAL: RAG(embedding_type=EmbeddingType.MISTRAL),
    }

    store: Dict[Tuple[str, str], BaseChatMessageHistory] = {}

    @staticmethod
    def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in ChatBot.store:
            ChatBot.store[(user_id, conversation_id)] = ChatMessageHistory()
        return ChatBot.store[(user_id, conversation_id)]

    @classmethod
    def format_document(cls, doc: Document) -> str:
        document_info = {
            "content": doc.page_content,
            "source": f"{doc.metadata['source_file']} / Page {doc.metadata['page_num']}",
        }
        return DOCUMENT_PROMPT.format(**document_info)

    @staticmethod
    def combine_documents(docs, document_separator="\n\n"):
        doc_strings = [ChatBot.format_document(doc) for doc in docs]
        context = document_separator.join(doc_strings)
        return context or "No relevant context found."

    @classmethod
    def get_vector_store(cls, embedding_type: EmbeddingType) -> VectorStore:
        return cls.VECTOR_STORES[embedding_type]

    @classmethod
    def construct_chain(cls, config: ChatBotConfig):
        vector_store = cls.get_vector_store(config.embedding_type)
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
            | config.get_condensation_model()
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
        #   chat_history: str                   #
        #   question: str                       #
        #########################################
        question_context = {
            "context": (
                itemgetter("standalone_question")
                | retriever
                | cls.combine_documents
            ),
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
            | config.get_chat_model()
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


    @overload
    @classmethod
    def get_chain(cls, model_family: Literal["GPT"], model: ChatBotConfig.OPENAI_MODELS): ...

    @overload
    @classmethod
    def get_chain(cls, model_family: Literal["Mistral"], model: ChatBotConfig.MISTRAL_MODELS): ...

    @overload
    @classmethod
    def get_chain(cls, model_family: Literal["Llama"], model: ChatBotConfig.LLAMA_MODELS): ...

    @classmethod
    def get_chain(cls, model_family: ChatBotConfig.MODEL_FAMILY, model: ChatBotConfig.MODELS):
        if not cls.chains:
            settings = [
                ChatBotConfig("GPT", model) for model in get_args(ChatBotConfig.OPENAI_MODELS)
            ] + [
                ChatBotConfig("Mistral", model) for model in get_args(ChatBotConfig.MISTRAL_MODELS)
            ] + [
                ChatBotConfig("Llama", model) for model in get_args(ChatBotConfig.LLAMA_MODELS)
            ]
            
            cls.chains = {
                (conf.embedding_type, conf.model): ChatBot.construct_chain(conf)
                for conf in settings
            } 


        settings = ChatBotConfig(model_family, model)
        return cls.chains[(settings.embedding_type, settings.model)]
        



if __name__ == "__main__":
    user_config = {
        "user_id": "user_id",
        "conversation_id": uuid.uuid4().hex,
    }

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
        chain = ChatBot.get_chain("GPT", "gpt-3.5-turbo")
        
        response = chain.stream({"question": question}, config={"configurable": user_config} | trace)

        for message in response:
            print(message, end="")


    # Test code 

    # models = [
    #     (model, ChatBot.get_chain("GPT", model)) for model in get_args(ChatBotConfig.OPENAI_MODELS)
    # ] + [
    #     (model, ChatBot.get_chain("Mistral", model)) for model in get_args(ChatBotConfig.MISTRAL_MODELS)
    # ] + [
    #     (model, ChatBot.get_chain("Llama", model)) for model in get_args(ChatBotConfig.LLAMA_MODELS)
    # ]

    # input_prompt = "What are the biggest improvements introducted in 5G? List 3 of the in a list!"

    # for model, chain in models:
    #     print(f"Model: {model}")
    #     response = chain.stream({"question": input_prompt}, config={"configurable": user_config} | trace)
    #     for message in response:
    #         print(message, end="")
    #     print("\n\n")

