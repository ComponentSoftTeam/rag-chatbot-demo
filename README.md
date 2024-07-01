# RAG Chatbot

## Introduction

We are aiming do make a chatbot that has access to our local files and can answer question about them. This aims to overcome the knowladge cutoff and halucination problems in chatbots.

We are using [Gradio](https://www.gradio.app/) to create a web interface for the chatbot and [LangChain](https://www.langchain.com/) to interact with different chat and embedding models.

## Getting Started

Clone the repository and install the requirements:
```bash
$ git clone https://github.com/ComponentSoftTeam/rag-chatbot-demo.git
$ cd rag-chatbot-demo
$ pip install -r requirements.txt
```

If you have conflicting dependencies, you can create a new environment and install the requirements there:
```bash
$ pythoh -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Copy the `.env.example` file to `.env` and fill in the required values

## Running the application

You can run the application from within the src directory.
```bash
$ cd src
$ python app.py
```

Then open your browser and navigate the link provided in the terminal.

> Note: The application is using python 3.10^
> Important: For the mistral embedding to work corretly it needs to load a tokenizer, for this the HF_TOKEN is set in the .env file, but you also need to visit the following link: [Mistral Huggingface](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) and agree to the terms and conditions. If you choose to not go with this option a heuristic solution will be used instead.

You can experiment with the jupyter notebook version of the application by loading it into a jupyter notebook environment.

# Development

Here are some documents that could be helpful for developers of a similar project

## RAG
- [Chroma persistent client](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [Adding sources](https://python.langchain.com/docs/use_cases/question_answering/sources)
- [Optimal Chunk size](https://www.youtube.com/watch?v=1bbDH3kyf9I)
- [Hybrid Search](https://www.youtube.com/watch?v=r2m9DbEmeqI)
- Prompt condensation

## Resources
- https://python.langchain.com/docs/use_cases/question_answering/quickstart
- https://www.aporia.com/learn/build-a-rag-chatbot/
- https://github.com/QuivrHQ/quivr





