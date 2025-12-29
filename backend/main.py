from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil

from .my_rag_pipeline import (
    create_vectorstore,
    get_answer_from_groq,
    pdfs_changed,
)

app = FastAPI(title="Multi-PDF RAG Bot")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)

vectorstore = None

# ================= MODELS =================
class Question(BaseModel):
    question: str

class DeletePDFs(BaseModel):
    filenames: list[str]

# ================= UPLOAD =================
@app.post("/upload-pdf")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global vectorstore

    for file in files:
        path = DATA_PATH / file.filename
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Only reset if PDFs really changed
    if pdfs_changed():
        vectorstore = None
        print("Active vectorstore invalidated (PDFs changed)")

    return {"status": "uploaded"}

# ================= DELETE =================
@app.post("/delete-pdf")
def delete_pdf(data: DeletePDFs):
    global vectorstore

    for name in data.filenames:
        file = DATA_PATH / name
        if file.exists():
            file.unlink()

    vectorstore = None
    pdfs_changed()  # update state

    return {"status": "deleted"}

# ================= ASK =================
@app.post("/ask")
def ask(data: Question):
    global vectorstore

    if not any(DATA_PATH.glob("*.pdf")):
        raise HTTPException(400, "No PDFs uploaded")

    if vectorstore is None:
        vectorstore = create_vectorstore()

    return {
        "answer": get_answer_from_groq(data.question, vectorstore)
    }

# ================= LIST =================
@app.get("/list-pdfs")
def list_pdfs():
    return {"pdfs": [f.name for f in DATA_PATH.glob("*.pdf")]}

@app.get("/")
def home():
    return {"status": "Multi-PDF RAG Bot running"}
