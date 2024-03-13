from tqdm import tqdm
from pypdf import PdfReader
from dotenv import find_dotenv, load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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

    openai_db = Chroma.from_texts(character_split_texts, OpenAIEmbeddings(), persist_directory="./openai_db")

    print(f"Uploaded {openai_db._collection.count()} records to Chroma")

if __name__ == "__main__":
    main()
