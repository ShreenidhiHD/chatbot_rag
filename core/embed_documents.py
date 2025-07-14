import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from .config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE_QA,
    CHUNK_OVERLAP_QA
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_vectorstore() -> Chroma:
    """
    Create and return ChromaDB vector store instance (same as products).
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )
    
    return vectorstore


def _parse_qa_file(file_path: Path) -> List[dict]:
    """
    Parse Q&A file and return list of Q&A pairs.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    qa_pairs = []
    sections = content.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        lines = section.split('\n')
        if len(lines) >= 2:
            question = lines[0].strip()
            answer = lines[1].strip()
            
            # Remove Q: and A: prefixes if present
            if question.startswith('Q:'):
                question = question[2:].strip()
            if answer.startswith('A:'):
                answer = answer[2:].strip()
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'full_text': f"Q: {question}\nA: {answer}"
            })
    
    return qa_pairs


def _create_document_chunks(qa_pairs: List[dict], source: str) -> List[Document]:
    """
    Create Document objects from Q&A pairs with chunking.
    """
    documents = []
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_QA,
        chunk_overlap=CHUNK_OVERLAP_QA,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for i, qa in enumerate(qa_pairs):
        # Create document for each Q&A pair
        doc = Document(
            page_content=qa['full_text'],
            metadata={
                "source": source,
                "qa_index": i,
                "document_type": "faq",
                "category": _get_category_from_source(source),
                "question": qa['question']
            }
        )
        
        # Check if chunking is needed
        if len(qa['full_text']) > CHUNK_SIZE_QA:
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = chunk_idx
                documents.append(chunk)
        else:
            # Add as single document
            doc.metadata["chunk_index"] = 0
            documents.append(doc)
    
    return documents


def _get_category_from_source(source: str) -> str:
    """
    Determine category based on source filename.
    """
    if "refund" in source.lower():
        return "refund"
    elif "general" in source.lower():
        return "general"
    elif "shipping" in source.lower():
        return "shipping"
    elif "policy" in source.lower():
        return "policy"
    else:
        return "general"


def embed_documents(file_name: Optional[str] = None) -> None:
    """
    Embed documents from files. Auto-scans all .txt files if no file_name provided.
    """
    documents_path = Path(DOCUMENTS_DIR)
    
    if not documents_path.exists():
        logger.error(f"Documents directory not found: {DOCUMENTS_DIR}")
        return
    
    # Get files to process
    if file_name:
        files_to_process = [documents_path / f"{file_name}.txt"]
        # Check if file exists
        if not files_to_process[0].exists():
            logger.error(f"File not found: {files_to_process[0]}")
            return
    else:
        # Auto-scan all .txt files
        files_to_process = list(documents_path.glob("*.txt"))
    
    if not files_to_process:
        logger.warning("No .txt files found to process")
        return
    
    logger.info(f"Processing {len(files_to_process)} document files")
    
    vectorstore = create_vectorstore()
    total_documents = 0
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            source = file_path.stem  # filename without extension
            
            # Parse Q&A pairs
            qa_pairs = _parse_qa_file(file_path)
            if not qa_pairs:
                logger.warning(f"No Q&A pairs found in {file_path.name}")
                continue
            
            # Create documents with chunking
            documents = _create_document_chunks(qa_pairs, source)
            
            # Add to vector store
            if documents:
                vectorstore.add_documents(documents)
                total_documents += len(documents)
                logger.info(f"Added {len(documents)} documents from {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    # Persist to disk
    vectorstore.persist()
    logger.info(f"Embedded and stored {total_documents} document chunks in ChromaDB")


def update_documents_by_source(source: str) -> None:
    """
    Update documents for a specific source (filename without extension).
    """
    vectorstore = create_vectorstore()
    
    # Delete existing documents from this source
    try:
        vectorstore.delete(where={"source": source})
        vectorstore.persist()
        logger.info(f"Deleted existing documents from source: {source}")
    except Exception as e:
        logger.warning(f"Error deleting existing documents from {source}: {e}")
    
    # Re-add documents from this source
    embed_documents(file_name=source)


def delete_documents_by_source(source: str) -> bool:
    """
    Delete all documents from a specific source.
    """
    vectorstore = create_vectorstore()
    
    try:
        vectorstore.delete(where={"source": source})
        vectorstore.persist()
        logger.info(f"Deleted all documents from source: {source}")
        return True
    except Exception as e:
        logger.error(f"Error deleting documents from {source}: {e}")
        return False


def update_all_documents() -> None:
    """
    Update all documents by deleting all document sources and re-adding.
    """
    vectorstore = create_vectorstore()
    
    # Delete all documents (keep products)
    try:
        vectorstore.delete(where={"document_type": "faq"})
        vectorstore.persist()
        logger.info("Deleted all existing documents")
    except Exception as e:
        logger.warning(f"Error deleting existing documents: {e}")
    
    # Re-add all documents
    embed_documents()


def list_document_sources() -> List[str]:
    """
    List all available document sources from the documents directory.
    """
    documents_path = Path(DOCUMENTS_DIR)
    
    if not documents_path.exists():
        logger.error(f"Documents directory not found: {DOCUMENTS_DIR}")
        return []
    
    txt_files = list(documents_path.glob("*.txt"))
    sources = [f.stem for f in txt_files]
    
    logger.info(f"Found {len(sources)} document sources: {sources}")
    return sources
