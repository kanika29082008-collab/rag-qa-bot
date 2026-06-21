# RAG Q&A Bot

A retrieval-augmented generation (RAG) API that answers questions about your own PDF documents — fully local, no API keys required.

## What this demonstrates

- End-to-end Generative AI pipeline: document loading → chunking → embeddings → vector search → LLM generation
- Using **LangChain** to orchestrate retrieval and generation
- Local vector search with **FAISS**
- Serving a RAG pipeline as a REST API with **FastAPI**

## Tech stack

- **LangChain** — RAG orchestration (built with LCEL, LangChain's composable expression language)
- **FAISS** — local vector store for similarity search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — generates embeddings
- **Hugging Face Transformers** (`google/flan-t5-base`) — generates answers, loaded and run directly via `AutoModelForSeq2SeqLM`
- **FastAPI** — REST API framework

## How it works

1. `ingest.py` loads a PDF, splits it into overlapping text chunks, embeds each chunk, and saves a FAISS index to disk (`vector_store/`).
2. `main.py` loads that index and the LLM at startup, then exposes a `/ask` endpoint: it embeds your question, retrieves the most relevant chunks, and asks the LLM to answer using only that context.

The RAG chain is built manually with LangChain's LCEL syntax (`retriever | prompt | llm`) rather than the older `RetrievalQA` helper, which was removed in recent LangChain versions.

## Setup

```bash
pip install -r requirements.txt
```

### Step 1 — Ingest your PDF

```bash
python ingest.py path/to/your/document.pdf
```

This creates a `vector_store/` folder containing the FAISS index. You only need to do this once per document (re-run it if the document changes).

### Step 2 — Start the API

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`. Open **http://localhost:8000/docs** for an interactive Swagger UI.

## Usage

### Example request

**POST** `/ask`

```json
{
  "question": "What programming languages does this person know?"
}
```

### Example response

```json
{
  "question": "What programming languages does this person know?",
  "answer": "Python, JavaScript, and SQL.",
  "sources": [
    { "content": "...relevant chunk of text from the PDF...", "page": 1 }
  ]
}
```

## Endpoints

| Method | Endpoint  | Description                                |
|--------|-----------|---------------------------------------------|
| GET    | `/`       | Health check / welcome message              |
| GET    | `/health` | Service, model, and vector store status     |
| POST   | `/ask`    | Ask a question about the ingested document  |

## Project structure

```
rag-qa-bot/
├── ingest.py           # Builds the FAISS vector store from a PDF
├── main.py             # FastAPI app exposing the /ask endpoint
├── requirements.txt    # Python dependencies
├── vector_store/       # Generated after running ingest.py (not committed)
└── README.md
```

## Notes

- Everything runs **locally on CPU** — no API keys, no external calls, no cost.
- `flan-t5-base` is instruction-tuned, so it follows the "answer using only this context" prompt well for a fully local, free model — though answers are simpler than a larger model like GPT-4 would give. This is a deliberate tradeoff for a free, fully local demo.
- The `sources` field in each response shows which chunks of the document were used to generate the answer — useful for verifying the model isn't hallucinating.

## License

This project is for educational/demo purposes.
