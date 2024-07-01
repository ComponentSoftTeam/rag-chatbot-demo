from enum import IntEnum

# Path to the collections
OPENAI_CHROMA_COLLECTION = "../chroma/openai"
STRANSFORMERS_CHROMA_COLLECTION = "../chroma/stransformers"
MISTRAL_CHROMA_COLLECTION = "../chroma/mistral"


# Embedding types
class EmbeddingType(IntEnum):
    OPEN_AI = 0
    SENTENCE_TRANSFORMER = 1
    MISTRAL = 2
