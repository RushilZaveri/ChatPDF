import os
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    return all_text

def ingest_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found - {pdf_path}")
        print(f"Make sure test.pdf is inside your rag-app folder")
        return
    print(f"Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    print("Loading embedding model... (first time takes 1 minute to download)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating vector store...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("Saved to faiss_index/")

if __name__ == "__main__":
    ingest_pdf("test.pdf")