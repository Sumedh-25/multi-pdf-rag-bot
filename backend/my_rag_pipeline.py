import os
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore"

DATA_PATH.mkdir(exist_ok=True)
VECTORSTORE_PATH.mkdir(exist_ok=True)

# ================= ENV =================
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("GROQ_API_KEY")

# ================= PDF STATE =================
STATE_FILE = VECTORSTORE_PATH / "pdf_state.txt"

def compute_pdf_state() -> str:
    """Create hash of all PDF filenames + sizes"""
    items = []
    for pdf in sorted(DATA_PATH.glob("*.pdf")):
        items.append(f"{pdf.name}:{pdf.stat().st_size}")
    raw = "|".join(items)
    return hashlib.md5(raw.encode()).hexdigest()

def pdfs_changed() -> bool:
    new_state = compute_pdf_state()
    if not STATE_FILE.exists():
        STATE_FILE.write_text(new_state)
        return True

    old_state = STATE_FILE.read_text()
    if old_state != new_state:
        STATE_FILE.write_text(new_state)
        return True

    return False

# ================= LOAD PDFs =================
def load_pdfs():
    docs = []
    for pdf in DATA_PATH.glob("*.pdf"):
        logger.info(f"Loading PDF: {pdf.name}")
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception:
            docs.extend(UnstructuredPDFLoader(str(pdf)).load())
    return docs

# ================= VECTORSTORE =================
def create_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = load_pdfs()
    if not documents:
        raise ValueError("No PDFs found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(VECTORSTORE_PATH)
    )

    logger.info(f"Vectorstore created with {len(chunks)} chunks")
    return vectordb

# ================= RAG =================
def get_answer_from_groq(query: str, vectordb):
    if not api_key:
        return "GROQ_API_KEY not set"

    docs = vectordb.similarity_search(query, k=30)
    if not docs:
        return "The requested information was not found in the provided documents."

    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    return llm.invoke(
        f"""
You are an academic assistant.
Answer ONLY using the context.

CONTEXT:
{context}

QUESTION:
{query}
"""
    ).content
