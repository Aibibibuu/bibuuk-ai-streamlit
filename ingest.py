import os
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# === Configuration ===
CHROMA_DIR = "chroma"
COLLECTION = "bibuuk_books"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 100

# === Helpers ===
def read_pdf(path: str) -> str:
    """Read all text from a PDF file."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=emb_fn)

# === Main ingestion ===
def ingest_folder(folder="data"):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    collection = get_collection()

    ids, docs, metas = [], [], []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        text = read_pdf(path)
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metas.append({"source": fname, "chunk": i})

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"✅ Ingested {len(docs)} chunks from {len(os.listdir(folder))} PDFs.")
    else:
        print("⚠️ No PDFs found in the 'data' folder.")

if __name__ == "__main__":
    ingest_folder("data")
