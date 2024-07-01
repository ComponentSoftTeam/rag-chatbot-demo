"""
This is a TUI for testing different models, and checkking if the current
vectordb is working properly.
"""

import uuid

from dotenv import load_dotenv, find_dotenv
from chatbot import ChatBot, ModelFamily, ModelName

load_dotenv(find_dotenv())

user_config = {
    "user_id": "user_id",
    "conversation_id": uuid.uuid4().hex,
}

# get chain
model_name = ModelName(ModelFamily.GPT, "gpt-3.5-turbo")
chain = ChatBot.get_chain(model_name)

print(f"UserId = {user_config['conversation_id']}")
print(f"Model = {model_name}")

while True:
    # print history
    history = ChatBot.get_session_history(
        user_config["user_id"], user_config["conversation_id"]
    )

    # get question
    question = input("\n\nEnter your question: ")
    print()
    if question == "exit":
        break

    response = chain.stream(
        {"question": question}, config={"configurable": user_config}
    )

    if response == "exit":
        break

    for message in response:
        print(message, end="")
