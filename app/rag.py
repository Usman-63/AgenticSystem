import os
import uuid
import logging
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.getenv("CHROMA_DIR", os.path.join("data", "chroma"))
EMB_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("KB_TOP_K", "3"))
SCORE_MODE = os.getenv("KB_SCORE_MODE", "similarity")
SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.3"))
logger = logging.getLogger(__name__)

def _company_dir(company_id: str) -> str:
    return os.path.join(BASE_DIR, f"company_{company_id}")

def _collection(company_id: str) -> str:
    return f"company_{company_id}"

def ingest_documents(company_id: str, file_paths: List[str]) -> Tuple[List[str], int]:
    os.makedirs(_company_dir(company_id), exist_ok=True)
    docs = []
    doc_ids = []
    for p in file_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
        else:
            loader = TextLoader(p, encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.update({"company_id": company_id, "source_path": p})
        docs.extend(loaded)
        doc_ids.append(str(uuid.uuid4()))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    vs = Chroma(collection_name=_collection(company_id), persist_directory=_company_dir(company_id), embedding_function=embeddings)
    vs.add_documents(chunks)
    vs.persist()
    return doc_ids, len(chunks)

def search_with_threshold(company_id: str, query: str):
    embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    vs = Chroma(collection_name=_collection(company_id), persist_directory=_company_dir(company_id), embedding_function=embeddings)
    results = vs.similarity_search_with_score(query, k=TOP_K)
    kept: List[Tuple] = []
    for doc, score in results:
        if SCORE_MODE == "distance":
            ok = score <= SCORE_THRESHOLD
        else:
            ok = score >= SCORE_THRESHOLD
        if ok:
            kept.append((doc, score))
    if not kept and results:
        kept.append((results[0][0], results[0][1]))
        logger.warning("KB threshold removed all results; using top-1")
    return kept
