from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from tqdm import tqdm

from config import (
    MISTRAL_CHROMA_COLLECTION,
    OPENAI_CHROMA_COLLECTION,
    STRANSFORMERS_CHROMA_COLLECTION,
)


def parse_pdf(path: str, pdf_name: str, page_offset: int = 0):
    """
    Parses a PDF file and extracts the text content from each page.

    Args:
        path (str): The path to the PDF file.
        pdf_name (str): The name of the PDF file to add to the metadata.
        page_offset (int, optional): The offset to be added to the page numbers (will be added to the metadata). Defaults to 0.

    Returns:
        list: A list of Document objects, each representing a page of the PDF.
    """

    reader = PdfReader(path)

    pdf_texts = []
    page_characters = [0] * (len(reader.pages) + 1)
    page_sum = 0
    for i, page in enumerate(tqdm(reader.pages)):
        page_text = page.extract_text().strip()
        pdf_texts.append(page_text)

        page_sum += len(page_text)
        page_characters[i] = int(page_sum)

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200
    )

    docs = []
    already_read_characters = 0
    curr_page = 0
    for chunk in character_splitter.split_text("".join(pdf_texts)):
        already_read_characters += len(chunk)
        for i, page in enumerate(page_characters):
            if already_read_characters < page:
                curr_page = i - 1
                break
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_file": str(pdf_name),
                    "page_num": curr_page + page_offset,
                },
            )
        )

    return docs


def main():
    load_dotenv(find_dotenv())

    path_book1 = "../data/5g_book.pdf"
    path_book2 = "../data/5g_book_2.pdf"

    docs = parse_pdf(
        path_book1,
        "https://digilib.stekom.ac.id/assets/dokumen/ebook/feb_6dc75f6bb1ff6ccaf3c3bc84d5bfb41cd71f701a_1652450470.pdf",
        page_offset=34,
    )
    docs += parse_pdf(
        path_book2,
        "https://sist.sathyabama.ac.in/sist_coursematerial/uploads/SECA3020.pdf",
        page_offset=2,
    )

    openai_db = Chroma.from_documents(
        docs, embedding=OpenAIEmbeddings(), persist_directory=OPENAI_CHROMA_COLLECTION
    )
    print(f"[OPENAI] Uploaded {openai_db._collection.count()} records to Chroma")

    stransformers_db = Chroma.from_documents(
        docs,
        embedding=HuggingFaceEmbeddings(),
        persist_directory=STRANSFORMERS_CHROMA_COLLECTION,
    )
    print(
        f"[SENTENCE-TRANSFORMERS] Uploaded {stransformers_db._collection.count()} records to Chroma"
    )

    mistral_db = Chroma.from_documents(
        docs,
        embedding=MistralAIEmbeddings(),
        persist_directory=MISTRAL_CHROMA_COLLECTION,
    )
    print(f"[MISTRAL] Uploaded {mistral_db._collection.count()} records to Chroma")


if __name__ == "__main__":
    main()
