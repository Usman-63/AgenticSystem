import os
import uuid
import logging
import threading
from typing import List, Tuple, Dict, Optional
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

# Helper functions (must be defined before cache functions that use them)
def _company_dir(company_id: str) -> str:
    return os.path.join(BASE_DIR, f"company_{company_id}")

def _collection(company_id: str) -> str:
    return f"company_{company_id}"

# Cached embeddings model (singleton pattern)
_embeddings_cache: Dict[str, SentenceTransformerEmbeddings] = {}
_embeddings_lock = threading.Lock()

def get_embeddings(model_name: str = EMB_MODEL) -> SentenceTransformerEmbeddings:
    """Get or create cached embeddings model instance"""
    if model_name not in _embeddings_cache:
        with _embeddings_lock:
            # Double-check after acquiring lock
            if model_name not in _embeddings_cache:
                logger.info(f"Loading embeddings model: {model_name}")
                _embeddings_cache[model_name] = SentenceTransformerEmbeddings(model_name=model_name)
                logger.info(f"Embeddings model loaded: {model_name}")
    return _embeddings_cache[model_name]

# Cached Chroma vector stores (per company)
_chroma_cache: Dict[str, Chroma] = {}
_chroma_lock = threading.Lock()

def get_chroma(company_id: str, embeddings: SentenceTransformerEmbeddings) -> Chroma:
    """Get or create cached Chroma vector store instance"""
    cache_key = f"{company_id}_{embeddings.model_name}"
    if cache_key not in _chroma_cache:
        with _chroma_lock:
            # Double-check after acquiring lock
            if cache_key not in _chroma_cache:
                logger.info(f"Creating Chroma vector store for company: {company_id}")
                _chroma_cache[cache_key] = Chroma(
                    collection_name=_collection(company_id),
                    persist_directory=_company_dir(company_id),
                    embedding_function=embeddings
                )
                logger.info(f"Chroma vector store created: {company_id}")
    return _chroma_cache[cache_key]

def ingest_documents(company_id: str, file_paths: List[str]) -> Tuple[Dict[str, str], int]:
    """
    Ingest documents into the vector store.
    Returns a mapping of filename -> doc_id and total chunks added.
    """
    os.makedirs(_company_dir(company_id), exist_ok=True)
    docs = []
    doc_id_map = {}  # filename -> doc_id mapping
    
    for p in file_paths:
        filename = os.path.basename(p)
        doc_id = str(uuid.uuid4())
        doc_id_map[filename] = doc_id
        
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
        else:
            loader = TextLoader(p, encoding="utf-8")
        loaded = loader.load()
        
        # Store doc_id in metadata for all documents from this file
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.update({
                "company_id": company_id,
                "source_path": p,
                "doc_id": doc_id,
                "filename": filename
            })
        docs.extend(loaded)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Ensure all chunks have doc_id metadata
    for chunk in chunks:
        if "doc_id" not in chunk.metadata:
            # Fallback: try to get from source_path
            source_path = chunk.metadata.get("source_path", "")
            filename = os.path.basename(source_path)
            if filename in doc_id_map:
                chunk.metadata["doc_id"] = doc_id_map[filename]
                chunk.metadata["filename"] = filename
    
    embeddings = get_embeddings(EMB_MODEL)
    vs = get_chroma(company_id, embeddings)
    vs.add_documents(chunks)
    vs.persist()
    logger.info(f"Ingested {len(file_paths)} documents, {len(chunks)} chunks, doc_id_map: {doc_id_map}")
    return doc_id_map, len(chunks)

def search_with_threshold(company_id: str, query: str):
    embeddings = get_embeddings(EMB_MODEL)
    vs = get_chroma(company_id, embeddings)
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

def delete_document(company_id: str, doc_id: str) -> bool:
    """
    Delete all chunks with the given doc_id from the vector store.
    Returns True if successful, False otherwise.
    """
    try:
        embeddings = get_embeddings(EMB_MODEL)
        vs = get_chroma(company_id, embeddings)
        
        # Get all documents with this doc_id
        # Chroma doesn't have a direct delete by metadata, so we need to:
        # 1. Get all document IDs with matching metadata
        # 2. Delete them
        
        # Get collection to access raw data
        collection = vs._collection
        
        # Query for documents with matching doc_id
        where_clause = {"doc_id": doc_id}
        results = collection.get(where=where_clause)
        
        if not results or not results.get("ids"):
            logger.warning(f"No documents found with doc_id={doc_id}")
            return False
        
        ids_to_delete = results["ids"]
        logger.info(f"Deleting {len(ids_to_delete)} chunks with doc_id={doc_id}")
        
        # Delete the documents
        collection.delete(ids=ids_to_delete)
        vs.persist()
        
        logger.info(f"Successfully deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document doc_id={doc_id}: {e}", exc_info=True)
        return False
