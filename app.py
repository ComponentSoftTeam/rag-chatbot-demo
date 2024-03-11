from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChatEntry:
    human_message: str
    bot_message: Optional[str]
    bot_rag: Optional[str]


@dataclass
class RagResponse:
    source: str
    content: str


class EmbeddingType(Enum):
    OPEN_AI = "openai"
    SENTENCE_TRANSFORMER = "sentence transformers"
    MISTRAL = "mistral"


@abstractmethod
def rag(
    prompt: str,
    max_files: int = 1,
    embedding_type: EmbeddingType = EmbeddingType.OPEN_AI
    ) -> List[RagResponse]:
    """Given a prompt, retuns the relevant file contents

    Args:
        prompt (str): _description_
        max_files (int): _description_

    Returns:
        List[RagResponse]: Sorted list based on relevance
    """



def generate(chat_history: List[ChatEntry]) -> str:
    return "".join(generate_streaming(chat_history))


def generate_streaming(chat_history: List[ChatEntry]) -> Iterator[str]:

    ### rag(msg = "Hello", k = 1)

    return [
        "Hello\n",
        "Hi!\n",
        "How are you?\n",
        "I am fine.\n",
        "What are you doing?\n",
        "I am generating text.\n",
        "What is your name?\n",
        "My name is ChatGPT.",
    ]


if __name__ == "__main__":
    chat_history = [
        ChatEntry(human_message="Hello", bot_message="Hi!", bot_rag="Hello"),
        ChatEntry(human_message="How are you?", bot_message="I am fine.", bot_rag="I am fine."),
        ChatEntry(
            human_message="What are you doing?", bot_message="I am generating text.", bot_rag="I am generating text."
        ),
        ChatEntry(human_message="What is your name?", bot_message=None, bot_rag=None),
    ]
    print(
        generate(chat_history)
    )  # 'Hello\nHi!\nHow are you?\nI am fine.\nWhat are you doing?\nI am generating text.\nWhat is your name?\nMy name is ChatGPT.
