from dotenv import load_dotenv
from app import ChatBot, ChatBotConfig
from typing import get_args

load_dotenv()

import uuid

# +
import random
import gradio as gr

username = "Mercedes"

modelfamilies_model_dict = {
    "GPT": get_args(ChatBotConfig.OPENAI_MODELS),
    "Mistral": get_args(ChatBotConfig.MISTRAL_MODELS),
    "Llama": get_args(ChatBotConfig.LLAMA_MODELS),
}

system_prompt = []
temperature = []
max_tokens = []
ChatBot.store = {}

def exec_prompt(chatbot, question, session_id, model_family = "Mistral", model="mistral-large"):

    question = question
    # get chain
    chain = ChatBot.get_chain(model_family=model_family, model=model)
    response = chain.invoke({"question": question}, config={"configurable": {"user_id": username, "conversation_id": session_id}})
    chat_history = ChatBot.get_session_history(username, session_id)

    history_pairs = []
    for msg in chat_history.messages:
        if msg.type == "human":
            history_pairs.append([msg.content, ""])
        elif msg.type == "ai":
            history_pairs[-1][1] = msg.content

    return history_pairs, "" 

def exec_prompt_streaming(chatbot, question, session_id, model_family = "Mistral", model="mistral-large"):

    question = question
    # get chain
    chain = ChatBot.get_chain(model_family=model_family, model=model)
    chat_history = ChatBot.get_session_history(username, session_id)
    response = chain.stream({"question": question}, config={"configurable": {"user_id": username, "conversation_id": session_id}})

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

with gr.Blocks(title="CompSoft") as demo:
    session_id = gr.Textbox(value = uuid.uuid4, interactive=False, visible=False)
    gr.Markdown("# Mercedes 5G RAG Demo")
    #system_prompt = gr.Textbox(label="System prompt", value="You are a helpful, harmless and honest assistant.")
    with gr.Row():
        modelfamily = gr.Dropdown(list(modelfamilies_model_dict.keys()), label="Model family", value="Mistral")
        model = gr.Dropdown(list(modelfamilies_model_dict["Mistral"]), label="Model", value="mistral-large")       
        """temperature = gr.Slider(label="Temperature:", minimum=0, maximum=2, value=1,
            info="LLM generation temperature")
        max_tokens = gr.Slider(label="Max tokens", minimum=100, maximum=2000, value=500, 
            info="Maximum number of generated tokens")"""
    with gr.Row():
        chatbot=gr.Chatbot(label="ComponentSoft_RAG", height=400, show_copy_button=True)
    with gr.Row():
        prompt = gr.Textbox(label="Question", value="What are the advantages of electrical motors?")
    with gr.Row():
        submit_btn_nostreaming = gr.Button("Answer")
        submit_btn_streaming = gr.Button("Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot])
        flag_btn = gr.Button("Flag")
    
    
    @modelfamily.change(inputs=modelfamily, outputs=[model])
    def update_modelfamily(modelfamily):
        model = list(modelfamilies_model_dict[modelfamily])
        return gr.Dropdown(choices=model, value=model[0], interactive=True)

    submit_btn_streaming.click(exec_prompt_streaming, inputs=[chatbot, prompt, session_id, modelfamily, model], outputs=[chatbot, prompt])
    submit_btn_nostreaming.click(exec_prompt, inputs=[chatbot, prompt, session_id, modelfamily, model], outputs=[chatbot, prompt])
    clear_btn.click(lambda session_id: ChatBot.del_session_history(username, session_id), [session_id], None, preprocess=False)

    callback.setup([modelfamily, model, chatbot], "flagged_data_points")
    flag_btn.click(lambda *args: callback.flag(args), [modelfamily, model, chatbot], None, preprocess=False)
    
    gr.Examples(
        ["What does electric drive means?", "What are the advantages of electrical motors?", "What are their main types?", "What is the difference between synchronous and asynchronous motors?",
         "What are DC choppers?", "What are their main types?", "What are the limitations of stator voltage control?", "What is the difference between rotor resistance and reactance?", "What is rotor power factor?"],
        prompt
    )

#demo.launch()
#demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=("Ericsson", "Torshamnsgatan21"), max_threads=20, show_error=True, state_session_capacity=20)
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=("Mercedes", "70372Stuttgart"), max_threads=20, show_error=True, favicon_path="data/favicon.ico", state_session_capacity=20)

# -


