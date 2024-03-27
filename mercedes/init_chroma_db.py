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

def parse_pdf(path: str, pdf_name: str, page_offset: int = 0):
    reader = PdfReader(path)

    pdf_texts = []
    page_characters = [0]*(len(reader.pages)+1)
    page_sum = 0
    for i, page in enumerate(tqdm(reader.pages)):
        page_text  = page.extract_text().strip()
        pdf_texts.append(page_text)

        page_sum += len(page_text)
        page_characters[i] = int(page_sum)

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = []
    already_read_characters = 0
    curr_page = 0
    for chunk in character_splitter.split_text(''.join(pdf_texts)):
        already_read_characters += len(chunk)
        for i, page in enumerate(page_characters):
            if already_read_characters < page:
                curr_page = i -1
                break
        docs.append(Document(page_content=chunk, metadata={"source_file": str(pdf_name), "page_num": curr_page+page_offset}))
                                               
    return docs

def main():
    load_dotenv(find_dotenv())

    path_book1 = "./data/S-Sedan Owners Manual.pdf"
    path_book2 = "./data/Electrical Drives_new.pdf"

    #docs = parse_pdf(path_book1, "https://www.mbusa.com/content/dam/mb-nafta/us/owners/manuals/2022/OperatorManuals/S-Sedan%20Owners%20Manual.pdf", page_offset=0)
    docs = parse_pdf(path_book2, "https://mrcet.com/downloads/digital_notes/EEE/31082020/Electrical%20Drives.pdf", page_offset=4)
    #docs += parse_pdf(path_book2, "https://sist.sathyabama.ac.in/sist_coursematerial/uploads/SECA3020.pdf", page_offset=2)

    openai_db = Chroma.from_documents(
        docs, 
        embedding=OpenAIEmbeddings(),
        persist_directory=OPENAI_CHROMA_COLLECTION
    )
    print(f"[OPENAI] Uploaded {openai_db._collection.count()} records to Chroma")
 

    stransformers_db = Chroma.from_documents(
        docs,
        embedding=SentenceTransformerEmbeddings(),
        persist_directory=STRANSFORMERS_CHROMA_COLLECTION
    )
    print(f"[SENTENCE-TRANSFORMERS] Uploaded {stransformers_db._collection.count()} records to Chroma")

    mistral_db = Chroma.from_documents(
        docs,
        embedding=MistralAIEmbeddings(),
        persist_directory=MISTRAL_CHROMA_COLLECTION
    )
    print(f"[MISTRAL] Uploaded {mistral_db._collection.count()} records to Chroma")


if __name__ == "__main__":
    main()
