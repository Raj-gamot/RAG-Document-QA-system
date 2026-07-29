from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from utils import (
    process_and_embed,
    save_to_chromadb,
    answer_query,
    get_chroma_client,
    get_supported_formats,
)

app = FastAPI(
    title="Mistral RAG System",
    description="Upload documents and ask questions with Retrieval-Augmented Generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = get_chroma_client()
COLLECTION_NAME = "documents"


class Question(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/")
def root():
    return {"status": "running", "service": "mistral-rag-system"}


@app.get("/supported-formats")
def supported_formats():
    return {"formats": get_supported_formats()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[-1].lower().replace(".", "")
    if ext not in get_supported_formats():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Supported: {', '.join(get_supported_formats())}",
        )

    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_and_embed(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable content found in document")

        save_to_chromadb(chunks, client, COLLECTION_NAME)
        return {"filename": safe_name, "total_chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query")
def query(request: Question):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return {
        "answer": answer_query(question, client, COLLECTION_NAME),
    }


@app.delete("/documents")
def clear_documents():
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(COLLECTION_NAME)
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
