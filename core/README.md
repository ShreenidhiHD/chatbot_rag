# Core Embedding Modules

This module provides functions to manage product and document embeddings in ChromaDB vector store using LangChain.

## Product Embedding Functions (embed_products.py)

### `embed_products()`
Initial embedding of all products from JSON file into ChromaDB. Use this for first-time setup.

### `embed_products_safe()`
Embed products with duplicate checking - only adds new products that don't already exist in the vector store.

### `create_vectorstore()`
Creates and returns a ChromaDB vector store instance with configured embedding model.

### `add_new_product(product_data: dict)`
Add a single new product to the vector store. Checks for duplicates before adding.

### `update_product_by_id(product_id: str)`
Update a specific product by its ID. Deletes the old version and adds the updated version from JSON.

### `delete_product_by_id(product_id: str)`
Delete a specific product from the vector store using its product ID.

### `update_all_products()`
Update entire product collection by deleting all existing products and re-adding from JSON file.

### `_create_product_document(product_data: dict)`
Helper function to convert product data dictionary into a LangChain Document object.

## Document Embedding Functions (embed_documents.py)

### `embed_documents(file_name=None)`
Embed documents from .txt files. Auto-scans all .txt files if no file_name provided, or processes specific file.

### `update_documents_by_source(source: str)`
Update documents for a specific source (filename without extension). Deletes existing and re-adds from file.

### `delete_documents_by_source(source: str)`
Delete all documents from a specific source.

### `update_all_documents()`
Update all documents by deleting all document sources and re-adding from files.

### `list_document_sources()`
List all available document sources from the documents directory.

### `_parse_qa_file(file_path: Path)`
Helper function to parse Q&A format files and extract question-answer pairs.

### `_create_document_chunks(qa_pairs: List[dict], source: str)`
Helper function to create Document objects from Q&A pairs with chunking when needed.

## Configuration

The modules use configuration from `config.py`:
- `PRODUCT_JSON_PATH`: Path to product catalog JSON file
- `DOCUMENTS_DIR`: Directory containing .txt document files
- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use
- `CHROMA_DB_DIR`: Directory to store ChromaDB data
- `CHUNK_SIZE_QA`: Chunk size for document splitting
- `CHUNK_OVERLAP_QA`: Overlap between chunks

## Usage

```python
# Product embedding
from llm.core.embed_products import embed_products, add_new_product, update_product_by_id

# Initial setup
embed_products()

# Add new product
new_product = {"product_id": "123", "product": "Coffee Beans", ...}
add_new_product(new_product)

# Update existing product
update_product_by_id("123")

# Document embedding
from llm.core.embed_documents import embed_documents, update_documents_by_source

# Process all .txt files
embed_documents()

# Process specific file
embed_documents("general_qa")

# Update only refund documents
update_documents_by_source("refund_policy")

# List available sources
from llm.core.embed_documents import list_document_sources
sources = list_document_sources()
```
update_product_by_id("123")
```

s