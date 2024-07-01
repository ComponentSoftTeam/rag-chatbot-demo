from enum import Enum
from operator import itemgetter

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import (
    ConfigurableFieldSpec,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_fireworks.chat_models import ChatFireworks
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI

from prompts import ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT
from rag_utils import RAG, EmbeddingType


class ModelFamily(Enum):
    """Represents the model families available for selection."""

    LLAMA = "Llama"
    GPT = "GPT"
    MISTRAL = "Mistral"
    GEMINI = "Gemini"
    CLAUDE = "Claude"


class ModelName(Enum):
    """
    Enum representing different model names.
    Each model name is associated with a model family and a model identifier.
    """

    LLAMA3_70B_PROMPTING = (ModelFamily.LLAMA, "llama-3-70b-prompting")
    LLAMA_3_70B_FIREFUNCTION_V2 = (ModelFamily.LLAMA, "Llama-3-70b-firefunction-v2")
    GPT_3_5_TURBO = (ModelFamily.GPT, "gpt-3.5-turbo")
    GPT_4O = (ModelFamily.GPT, "gpt-4o")
    GPT_4_TURBO = (ModelFamily.GPT, "gpt-4-turbo")
    MISTRAL_LARGE = (ModelFamily.MISTRAL, "mistral-large")
    OPEN_MIXTRAL_8X22B = (ModelFamily.MISTRAL, "open-mixtral-8x22b")
    MISTRAL_SMALL = (ModelFamily.MISTRAL, "mistral-small")
    GEMINI_1_5_FLASH = (ModelFamily.GEMINI, "gemini-1.5-flash")
    GEMINI_1_5_PRO = (ModelFamily.GEMINI, "gemini-1.5-pro")
    CLAUDE_3_HAIKU = (ModelFamily.CLAUDE, "claude-3-haiku")
    CLAUDE_3_SONNET = (ModelFamily.CLAUDE, "claude-3-sonnet")
    CLAUDE_3_OPUS = (ModelFamily.CLAUDE, "claude-3-opus")


def get_llm(
    model_name: ModelName, temperature: float, max_new_tokens: int
) -> BaseChatModel:
    """
    Returns a chat model based on the specified model name, temperature, and maximum number of new tokens.

    Args:
        model_name (ModelName): The name of the model to use.
        temperature (float): The temperature parameter for generating responses, [0, 2].
        max_new_tokens (int): The maximum number of new tokens to generate in the response.

    Returns:
        BaseChatModel: The chat model based on the specified parameters.

    Raises:
        RuntimeError: If an invalid model name is provided.
    """

    match model_name:
        case ModelName.LLAMA3_70B_PROMPTING:
            return ChatFireworks(
                name="accounts/fireworks/models/llama-v3-70b-instruct",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.LLAMA_3_70B_FIREFUNCTION_V2:
            return ChatFireworks(
                name="accounts/fireworks/models/firefunction-v1",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.GPT_3_5_TURBO:
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.GPT_4O:
            return ChatOpenAI(
                model="gpt-4o",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.GPT_4_TURBO:
            return ChatOpenAI(
                model="gpt-4-turbo",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.MISTRAL_LARGE:
            return ChatMistralAI(
                name="mistral-large-latest",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.OPEN_MIXTRAL_8X22B:
            return ChatMistralAI(
                name="open-mixtral-8x22b",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.MISTRAL_SMALL:
            return ChatMistralAI(
                name="mistral-small-latest",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.GEMINI_1_5_FLASH:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.GEMINI_1_5_PRO:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.CLAUDE_3_HAIKU:
            return ChatAnthropic(
                model_name="claude-3-haiku-20240307",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.CLAUDE_3_SONNET:
            return ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case ModelName.CLAUDE_3_OPUS:
            return ChatAnthropic(
                model_name="claude-3-opus-20240229",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        case _:
            raise RuntimeError(f"Invalid input model_name: {model_name}")


class ChatBot:
    VECTOR_STORES: dict[EmbeddingType, VectorStore] | None = None

    store: dict[tuple[str, str], BaseChatMessageHistory] = {}

    @staticmethod
    def get_models_by_families() -> dict[str, list[str]]:
        # Model names by families
        model_by_families: dict[str, list[str]] = {
            family.value: [] for family in ModelFamily
        }
        for model in ModelName:
            family, name = model.value
            model_by_families[family.value].append(name)

        return model_by_families

    @staticmethod
    def get_condensation_model(
        model_name: ModelName, max_new_tokens: int
    ) -> BaseChatModel:
        return get_llm(
            model_name=model_name, max_new_tokens=max_new_tokens, temperature=0.02
        )

    @staticmethod
    def get_chat_model(model_name: ModelName, max_new_tokens: int) -> BaseChatModel:
        return get_llm(
            model_name=model_name, max_new_tokens=max_new_tokens, temperature=0.7
        )

    @staticmethod
    def get_embedding_type(model_family: ModelFamily) -> EmbeddingType:
        match model_family:
            case ModelFamily.LLAMA:
                return EmbeddingType.SENTENCE_TRANSFORMER
            case ModelFamily.GPT:
                return EmbeddingType.OPEN_AI
            case ModelFamily.MISTRAL:
                return EmbeddingType.MISTRAL
            case ModelFamily.GEMINI:
                return EmbeddingType.OPEN_AI
            case ModelFamily.CLAUDE:
                return EmbeddingType.OPEN_AI

    @staticmethod
    def get_session_history(
        user_id: str, conversation_id: str
    ) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in ChatBot.store:
            ChatBot.store[(user_id, conversation_id)] = ChatMessageHistory()
        return ChatBot.store[(user_id, conversation_id)]

    @staticmethod
    def del_session_history(user_id: str, conversation_id: str) -> None:
        if (user_id, conversation_id) in ChatBot.store:
            del ChatBot.store[(user_id, conversation_id)]

    @classmethod
    def format_document(cls, doc: Document) -> str:
        document_info = {
            "content": doc.page_content,
            "source": f"{doc.metadata['source_file']} / Page {doc.metadata['page_num']}",
        }
        return DOCUMENT_PROMPT.format(**document_info)

    @staticmethod
    def combine_documents(docs: list[Document], document_separator="\n\n") -> str:
        doc_strings = [ChatBot.format_document(doc) for doc in docs]
        context = document_separator.join(doc_strings)
        return context or "No relevant context found."

    @classmethod
    def get_vector_store(cls, embedding_type: EmbeddingType) -> VectorStore:
        if not cls.VECTOR_STORES:
            cls.VECTOR_STORES = {
                EmbeddingType.OPEN_AI: RAG(embedding_type=EmbeddingType.OPEN_AI),
                EmbeddingType.SENTENCE_TRANSFORMER: RAG(
                    embedding_type=EmbeddingType.SENTENCE_TRANSFORMER
                ),
                EmbeddingType.MISTRAL: RAG(embedding_type=EmbeddingType.MISTRAL),
            }

        return cls.VECTOR_STORES[embedding_type]

    @classmethod
    def construct_chain(cls, model_name: ModelName) -> RunnableLambda:
        model_family, _ = model_name.value
        embedding_type = cls.get_embedding_type(model_family)
        vector_store = cls.get_vector_store(embedding_type)
        retriever = vector_store.as_retriever()

        #####################################x
        # INPUTS:                            #
        #   chat_history: ChatMessageHistory #
        #   question: str                    #
        # OUTPUTS:                           #
        #   standalone_question: str         #
        #   chat_history: ChatMessageHistory #
        #   question: str                    #
        #####################################x
        standalone_question = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | cls.get_condensation_model(model_name, max_new_tokens=256)
            | StrOutputParser(),
            question=lambda x: x["question"],
            chat_history=lambda x: x["chat_history"],
        )

        #########################################
        # INPUTS:                               #
        #   standalone_question: str            #
        #   chat_history: ChatMessageHistory    #
        #   question: str                       #
        # OUTPUTS:                              #
        #   context: str                        #
        #   chat_history: ChatMessageHistory    #
        #   question: str                       #
        #########################################
        question_context = {
            "context": (
                itemgetter("standalone_question") | retriever | cls.combine_documents
            ),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
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
            | cls.get_chat_model(model_name, max_new_tokens=512)
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

    chains: dict[ModelName, RunnableLambda] | None = None

    @classmethod
    def get_chain(cls, model_name: ModelName):
        if not cls.chains:
            cls.chains = {
                m_name: ChatBot.construct_chain(m_name) for m_name in ModelName
            }

        return cls.chains[model_name]
