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

from rag_utils import RAG, EmbeddingType, RagResponse

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

# trace = {
#     "callbacks": [
#         CallbackHandler(
#             secret_key="sk-lf-de74539a-7177-49a8-9c3e-50492e90d1a9",
#             public_key="pk-lf-71d6d1c4-7a5b-4f46-b75c-cd8dad7e35ff",
#             host="http://localhost:3000",
#         )
#     ]
# }

# while True:
#     # print history
#     history = ChatBot.get_session_history(user_config["user_id"], user_config["conversation_id"])
#     print(history)
#
#     # get question
#     question = input("Enter your question: ")
#
#     # get chain
#     chain = ChatBot.get_chain(embedding_type=EmbeddingType.OPEN_AI, model_type="openai")
#     response = chain.invoke({"question": question}, config={"configurable": user_config})

# while True:
#     # print history
#     for user, message in ChatBot.get_session_history(user_config["user_id"], user_config["conversation_id"]):
#         print(f"user: {user}, message: {message}")
#
#     # get question
#     question = input("Enter your question: ")
#     if question == "exit": break
#     # get chain
#     chain = ChatBot.get_chain(embedding_type=EmbeddingType.OPEN_AI, model_type="openai")
#     response = chain.invoke({"question": question}, config={"configurable": user_config})

import gradio as gr

modelfamilies_model_dict = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
    "Mistral": ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"],
    "Llama": ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"],
}

system_prompt = []
temperature = []
max_tokens = []
ChatBot.store = {}

def exec_prompt(chatbot, question, model_family = "Mistral", model="mistral-large"):

    question = question
    # get chain
    chain = ChatBot.get_chain(embedding_type=EmbeddingType.OPEN_AI, model_type="openai")
    #response = chain.invoke({"question": question}, config={"configurable": user_config} | trace)
    response = chain.invoke({"question": question}, config={"configurable": user_config})
    ChatBot_history = ChatBot.get_session_history(user_config["user_id"], user_config["conversation_id"])
    history = []
    for _, chat in ChatBot_history:
        """print(_)
        print(chat)
        print("\n\n\n")"""
        for i, chat in enumerate(chat):
            print(f"chat = {chat}")
            if i % 2 == 0:
                chat_str = str(chat)
                new_round = [chat_str[len("content='"):-1]]
                history.append(new_round)
                print(f"i={i}, text={new_round}, history={history}")
            else:
                chat_str = str(chat)
                new_round = new_round.append(chat_str[len("content='"):-1])

    #print(history)
    return history, "" 

"""def exec_prompt_streaming(chatbot, prompt, model_family = "Mistral", model="mistral-large"):
    Prompt.set_system_prompt(system_prompt)
    Prompt.set_temperature(temperature)
    Prompt.set_max_tokens(max_tokens)
    Prompt.set_model(model_family, model)
    
    chat_history = chat_history or []
    chat_history.append([prompt, ""])
    stream = Prompt.exec_streaming(chat_history)
    for new_token in stream:
        if new_token is not None:
            chat_history[-1][1] += str(new_token)
        yield chat_history, "" """

gr.close_all()

callback = gr.CSVLogger()

with gr.Blocks(title="CompSoft") as demo:
    gr.Markdown("# Component Soft 5G RAG Demo")
    #system_prompt = gr.Textbox(label="System prompt", value="You are a helpful, harmless and honest assistant.")
    with gr.Row():
        modelfamily = gr.Dropdown(list(modelfamilies_model_dict.keys()), label="Model family", value="OpenAI")
        model = gr.Dropdown(list(modelfamilies_model_dict["OpenAI"]), label="Model", value="gpt-3.5-turbo")       
        """temperature = gr.Slider(label="Temperature:", minimum=0, maximum=2, value=1,
            info="LLM generation temperature")
        max_tokens = gr.Slider(label="Max tokens", minimum=100, maximum=2000, value=500, 
            info="Maximum number of generated tokens")"""
    with gr.Row():
        #chatbot=gr.Textbox(label="CompSoft_5G_RAG", lines=10, max_lines=20, show_copy_button=True)
        chatbot=gr.Chatbot(label="CompSoft_5G_RAG", height=400, show_copy_button=True)
    with gr.Row():
        prompt = gr.Textbox(label="Question", value="What is 5G?")
    with gr.Row():
        submit_btn_nostreaming = gr.Button("Answer")
        #submit_btn_streaming = gr.Button("Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot])
        #flag_btn = gr.Button("Flag")
    
    
    @modelfamily.change(inputs=modelfamily, outputs=[model])
    def update_modelfamily(modelfamily):
        model = list(modelfamilies_model_dict[modelfamily])
        return gr.Dropdown(choices=model, value=model[0], interactive=True)

    #submit_btn_streaming.click(exec_prompt_streaming, inputs=[chatbot, prompt, modelfamily, model], outputs=[chatbot, prompt])
    submit_btn_nostreaming.click(exec_prompt, inputs=[chatbot, prompt, modelfamily, model], outputs=[chatbot, prompt])

    #callback.setup([system_prompt, modelfamily, model, temperature, max_tokens, chatbot], "flagged_data_points")
    #flag_btn.click(lambda *args: callback.flag(args), [system_prompt, modelfamily, model, temperature, max_tokens, chatbot], None, preprocess=False)
    
    gr.Examples(
        ["What is 5G?", "What are the main adventages of 5G compared to 4G?", "What frequencies does 5G use?", "What is OFDMA?", 
         "Which organisations are responsible for the standardization of 5G?"],
        prompt
    )

#demo.launch()
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=("Ericsson", "Torshamnsgatan21"), max_threads=20, show_error=True, favicon_path="/home/rconsole/GIT/AI-434/source/labfiles/data/favicon.ico", state_session_capacity=20)



