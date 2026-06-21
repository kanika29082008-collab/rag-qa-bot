"""
RAG-based Q&A API - answers questions about a PDF document using
retrieval-augmented generation (LangChain + FAISS + a local Hugging Face LLM).

Run with:
    uvicorn main:app --reload

Then open http://localhost:8000/docs for an interactive test UI.

NOTE: you must run `python ingest.py your_file.pdf` first to create the
vector_store/ directory before starting this API.
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-qa-api")

app = FastAPI(
    title="RAG Q&A API",
    description="Ask questions about a PDF document using retrieval-augmented generation.",
    version="1.0.0",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
VECTOR_STORE_DIR = "vector_store"
TOP_K = 3

qa_chain = None
retriever = None

if not os.path.exists(VECTOR_STORE_DIR):
    logger.warning(
        f"No vector store found at ./{VECTOR_STORE_DIR}/. "
        f"Run `python ingest.py your_file.pdf` before using /ask."
    )
else:
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Loading FAISS vector store...")
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    logger.info(f"Loading LLM: {LLM_MODEL} ...")
    # transformers 5.x rejects "text2text-generation" as a pipeline() task
    # name even when a model/tokenizer is passed in directly, so we call
    # the model ourselves instead of going through pipeline() at all.
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    def run_flan_t5(prompt_text: str) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    llm = RunnableLambda(run_flan_t5)

    prompt = PromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | (lambda prompt_value: prompt_value.to_string())
        | llm
        | StrOutputParser()
    )
    logger.info("RAG pipeline ready.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the document")


class SourceChunk(BaseModel):
    content: str
    page: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Q&A API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "healthy" if qa_chain else "vector store not loaded",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if qa_chain is None or retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Run `python ingest.py your_file.pdf` first.",
        )

    try:
        answer = qa_chain.invoke(request.question).strip()
        retrieved_docs = retriever.invoke(request.question)
        sources = [
            SourceChunk(content=doc.page_content, page=doc.metadata.get("page"))
            for doc in retrieved_docs
        ]
        return AskResponse(question=request.question, answer=answer, sources=sources)
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=str(e))