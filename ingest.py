"""
ingest.py — Build a local FAISS vector store from a PDF document.

This is "Step 1" of the RAG pipeline: it reads a PDF, splits it into
overlapping chunks, embeds each chunk with a sentence-transformer model,
and saves the resulting FAISS index to disk so the API (main.py) can
load it later without re-processing the PDF every time.

Run with:
    python ingest.py path/to/your/document.pdf
"""

import sys
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap so context isn't lost at chunk boundaries
VECTOR_STORE_DIR = "vector_store"


def build_vector_store(pdf_path: str):
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} page(s).")

    logger.info("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Embedding chunks and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(VECTOR_STORE_DIR)
    logger.info(f"Vector store saved to ./{VECTOR_STORE_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    build_vector_store(pdf_path)
