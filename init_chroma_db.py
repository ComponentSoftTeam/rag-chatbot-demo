from tqdm import tqdm
from PyPDF2 import PdfReader
from dotenv import find_dotenv, load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import OPENAI_CHROMA_COLLECTION, STRANSFORMERS_CHROMA_COLLECTION, MISTRAL_CHROMA_COLLECTION

def parse_pdf(path, pdf_name):
    def find_text_in_pdf(target_text):    
        for i, txt in enumerate(page_texts):
            if target_text in txt:
                return i + 1
        return None
    
    reader = PdfReader(path)
    pdf_texts = [p.extract_text().strip() for p in tqdm(reader.pages)]

    num_pages = len(reader.pages)
    page_texts = [reader.pages[i].extract_text() for i in range(num_pages)]
    
    
    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = [Document(page_content=chunk, file_source=pdf_name, page_number=find_text_in_pdf(chunk)) for chunk in character_splitter.split_text(''.join(pdf_texts))]

    return docs

def main():
    load_dotenv(find_dotenv())

    path_book1 = "./data/5g_book.pdf"
    path_book2 = "./data/5g_book_2.pdf"

    docs = parse_pdf(path_book1, "5g_book.pdf")
    docs += parse_pdf(path_book2, "5g_book_2.pdf")

    openai_db = Chroma.from_documents(
        docs, 
        embedding=OpenAIEmbeddings(),
        persist_directory=OPENAI_CHROMA_COLLECTION
    )
    print(f"Uploaded {openai_db._collection.count()} records to Chroma")

    return 

    stransformers_db = Chroma.from_documents(
        docs,
        embedding=SentenceTransformerEmbeddings(),
        persist_directory=STRANSFORMERS_CHROMA_COLLECTION
    )
    print(f"Uploaded {stransformers_db._collection.count()} records to Chroma")

    mistral_db = Chroma.from_documents(
        docs,
        embedding=MistralAIEmbeddings(),
        persist_directory=MISTRAL_CHROMA_COLLECTION
    )
    print(f"Uploaded {mistral_db._collection.count()} records to Chroma")


if __name__ == "__main__":
    main()
