from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings

import chromadb
import os

from dotenv import load_dotenv, find_dotenv


def read_pdf(path):
    reader = PdfReader(path)
    pdf_texts = [p.extract_text().strip() for p in tqdm(reader.pages)]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    return pdf_texts

def split_pdf_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    return character_splitter.split_text(''.join(texts))

def main():
    load_dotenv(find_dotenv())

    path_book1 = "./data/5g_book.pdf"
    path_book2 = "./data/5g_book_2.pdf"

    pdf_read_texts = read_pdf(path_book1)
    pdf_read_texts += read_pdf(path_book2)

    character_split_texts = split_pdf_texts(pdf_read_texts)

    chroma_db = Chroma.from_texts(character_split_texts, OpenAIEmbeddings(), persist_directory="./chroma_db")

    print(f"Uploaded {chroma_db._collection.count()} records to Chroma")

if __name__ == "__main__":
    main()
