import os
import uuid
from typing import Iterable

import gradio as gr
from dotenv import load_dotenv, find_dotenv

from chatbot import ChatBot, ModelFamily, ModelName

# The .env file is expected to be in the root directory of the project
load_dotenv(find_dotenv())

GRADIO_USER = os.environ["GRADIO_USER"]
GRADIO_PASSWORD = os.environ["GRADIO_PASSWORD"]

model_by_families = ChatBot.get_models_by_families()


def exec_prompt(
    question: str, session_id: str, model_family: str, model_name: str
) -> tuple[list[list[str]], str]:
    """
    Performs the retrieval and feeds the relevant documents to the model as the context.

    Two LLM calls are made, one to condense the prompt and create a standalone question
    from the history and question to be used with the retriever, and the other to answer
    the question based on the context.

    Args:
        question (str): The question to be asked.
        session_id (str): The ID of the chat session. This defines the history.
        model_family (str): The model family.
        model_name (str): The name of the model.

    Returns:
        tuple[list[list[str]], str]: A tuple containing the history pairs and an empty string.
    """

    model_family_kind = ModelFamily(model_family)
    model_name_kind = ModelName((model_family_kind, model_name))
    question = question or "I have no question"

    chain = ChatBot.get_chain(model_name_kind)

    # Automatically adds to history
    _ = chain.invoke(
        input={
            "question": question,
        },
        config={
            "configurable": {
                "user_id": GRADIO_USER,
                "conversation_id": session_id,
            }
        },
    )

    chat_history = ChatBot.get_session_history(GRADIO_USER, session_id)

    history_pairs = []
    for msg in chat_history.messages:
        if msg.type == "human":
            history_pairs.append([msg.content, ""])
        elif msg.type == "ai":
            history_pairs[-1][1] = msg.content

    return history_pairs, ""


def exec_prompt_streaming(
    question: str, session_id: str, model_family: str, model_name: str
) -> Iterable[tuple[list[list[str]], str]]:
    """
    Performs the retrieval and feeds the relevant documents to the model as the context.

    Two LLM calls are made, one to condense the prompt and create a standalone question
    from the history and question to be used with the retriever, and the other to answer
    the question based on the context.
    Out of the two calls, the second one is streamed back to the caller

    Args:
        question (str): The question to be asked.
        session_id (str): The ID of the chat session. This defines the history.
        model_family (str): The model family.
        model_name (str): The name of the model.
    
    Yields:
        Iterable[tuple[list[list[str]], str]]: A generator that yields a tuple containing the history pairs and an empty string.
    """

    model_family_kind = ModelFamily(model_family)
    model_name_kind = ModelName((model_family_kind, model_name))
    question = question or "I have no question"

    chain = ChatBot.get_chain(model_name_kind)

    chat_history = ChatBot.get_session_history(GRADIO_USER, session_id)
    response = chain.stream(
        input={
            "question": question,
        },
        config={
            "configurable": {
                "user_id": GRADIO_USER,
                "conversation_id": session_id,
            }
        },
    )

    history_pairs = []

    for msg in chat_history.messages:
        if msg.type == "human":
            history_pairs.append([msg.content, ""])
        elif msg.type == "ai":
            history_pairs[-1][1] = msg.content

    history_pairs.append([question, ""])
    for res in response:
        if res is not None:
            history_pairs[-1][1] += res
        yield history_pairs, ""


gr.close_all()

callback = gr.CSVLogger()


def save_datapoint(*args):
    callback.flag(args)  # type: ignore
    gr.Info("Data point flagged for review.")


with gr.Blocks(title="CompSoft") as demo:
    session_id = gr.Textbox(
        value=uuid.uuid4,
        interactive=False,
        visible=False,
    )

    # Gradio UI
    gr.Markdown("# Component Soft 5G RAG Demo")

    with gr.Row():
        model_family = gr.Dropdown(
            choices=list(model_by_families.keys()),
            label="Model family",
            value="Mistral",
        )

        model_name = gr.Dropdown(
            list(model_by_families[model_family.value]),
            label="Model",
            value="mistral-large",
        )

    with gr.Row():
        chatbot = gr.Chatbot(
            label="ComponentSoft_5G_RAG",
            height=400,
            show_copy_button=True,
        )

    with gr.Row():
        prompt = gr.Textbox(label="Question", value="What is 5G?")

    with gr.Row():
        submit_btn_nostreaming = gr.Button(value="Answer")
        submit_btn_streaming = gr.Button(value="Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot])
        flag_btn = gr.Button("Flag")

    gr.Examples(
        [
            "What is 5G?",
            "What are its main adventages compared to 4G?",
            "What frequencies does it use?",
            "Which organisations are responsible for its standardization?",
            "What is OFDMA?",
            "What is the difference between OFDMA and OFDM?",
            "What are the main components of 5G core networks?",
            "What were the design principles of Massive MTC?",
        ],
        prompt,
    )

    model_family.change(
        fn=lambda family: gr.Dropdown(
            choices=list(model_by_families[family]),
            label="Model",
            value=model_by_families[family][0],
            interactive=True,
        ),
        inputs=model_family,
        outputs=model_name,
    )

    submit_btn_streaming.click(
        exec_prompt_streaming,
        inputs=[prompt, session_id, model_family, model_name],
        outputs=[chatbot, prompt],
    )

    submit_btn_nostreaming.click(
        exec_prompt,
        inputs=[prompt, session_id, model_family, model_name],
        outputs=[chatbot, prompt],
    )

    clear_btn.click(
        fn=lambda session_id: ChatBot.del_session_history(GRADIO_USER, session_id),
        inputs=session_id,
        preprocess=False,
    )

    flag_btn.click(
        fn=save_datapoint,  # type: ignore
        inputs=[model_family, model_name, chatbot],
        preprocess=False,
    )

    callback.setup([model_family, model_name, chatbot], "../flagged_data_points")

# demo.launch()
# demo.launch(share=True)
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=(GRADIO_USER, GRADIO_PASSWORD), max_threads=20, show_error=True, favicon_path="../data/favicon.ico", state_session_capacity=20)
