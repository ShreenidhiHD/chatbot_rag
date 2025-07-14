# core/config.py

# Paths
PRODUCT_JSON_PATH = "core/data/products/product_catalog.json"
DOCUMENTS_DIR = "core/data/documents"
CHROMA_DB_DIR = "data/vectorstore/products"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# QA Document chunking (same as products for now)
CHUNK_SIZE_QA = CHUNK_SIZE
CHUNK_OVERLAP_QA = CHUNK_OVERLAP

# ChromaDB collection
CHROMA_COLLECTION_NAME = "product_data"

# Token limits (optional for LLMs later)
LLM_CONTEXT_SIZE = 2048

# LLM Configuration
LOCAL_LLM_MODEL_PATH = "~/Desktop/deve/llm_models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LOCAL_LLM_MODEL_NAME = "mistral-7b-instruct-v0.2"
DEFAULT_LLM_PROVIDER = "local"  # Options: "local", "openai", "gemini"

# API Keys (set these in environment variables)
OPENAI_API_KEY = None  # Set via environment variable
GEMINI_API_KEY = None  # Set via environment variable

# LLM Parameters
MAX_TOKENS = 1000  # Increased from 500 to prevent response cutoff
TEMPERATURE = 0.7
LLM_THREADS = 4
