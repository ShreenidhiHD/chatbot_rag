import json
import logging
from typing import List
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
from .config import PRODUCT_JSON_PATH, EMBEDDING_MODEL_NAME, CHROMA_DB_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_products() -> None:
    """
    Load products from JSON and embed them into ChromaDB vector store.
    """
    # Load products from JSON
    with open(PRODUCT_JSON_PATH, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # Convert to documents
    documents = []
    for row in tqdm(products, desc="Converting to documents"):
        content = (
            f"Product: {row['product']}\n"
            f"Group: {row['product_group']}\n"
            f"Category: {row['product_category']}\n"
            f"Type: {row['product_type']}\n"
            f"Description: {row['product_description']}\n"
            f"Size: {row['unit_of_measure']}\n"
            f"Wholesale Price: {row['current_wholesale_price']}\n"
            f"Retail Price: {row['current_retail_price']}\n"
            f"Tax Exempt: {row['tax_exempt_yn']}\n"
            f"Promo: {row['promo_yn']}\n"
            f"New Product: {row['new_product_yn']}"
        )

        metadata = {
            "product_id": str(row["product_id"]),
            "product": row["product"],
            "category": row["product_category"],
            "type": "product",
            "promo": row["promo_yn"],
            "new": row["new_product_yn"],
            "source": "product_json"
        }

        documents.append(Document(page_content=content, metadata=metadata))
    
    vectorstore = create_vectorstore()
    vectorstore.add_documents(documents)
    vectorstore.persist() 
    
    logger.info(f"Embedded and stored {len(documents)} product documents in ChromaDB")


def create_vectorstore() -> Chroma:
    """
    Create and return ChromaDB vector store instance.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        collection_name="product_data",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )
    
    return vectorstore


def _create_product_document(product_data: dict) -> Document:
    """
    Helper function to create a Document from product data.
    """
    content = (
        f"Product: {product_data['product']}\n"
        f"Group: {product_data['product_group']}\n"
        f"Category: {product_data['product_category']}\n"
        f"Type: {product_data['product_type']}\n"
        f"Description: {product_data['product_description']}\n"
        f"Size: {product_data['unit_of_measure']}\n"
        f"Wholesale Price: {product_data['current_wholesale_price']}\n"
        f"Retail Price: {product_data['current_retail_price']}\n"
        f"Tax Exempt: {product_data['tax_exempt_yn']}\n"
        f"Promo: {product_data['promo_yn']}\n"
        f"New Product: {product_data['new_product_yn']}"
    )

    metadata = {
        "product_id": str(product_data["product_id"]),
        "product": product_data["product"],
        "category": product_data["product_category"],
        "type": "product",
        "promo": product_data["promo_yn"],
        "new": product_data["new_product_yn"],
        "source": "product_json"
    }

    return Document(page_content=content, metadata=metadata)


def delete_product_by_id(product_id: str) -> bool:
    """
    Delete a specific product by product_id.
    """
    vectorstore = create_vectorstore()
    
    try:
        # Delete by product_id filter
        vectorstore.delete(where={"product_id": str(product_id)})
        vectorstore.persist()
        logger.info(f"Deleted product with ID: {product_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting product {product_id}: {e}")
        return False


def add_new_product(product_data: dict) -> bool:
    """
    Add a single new product, checking for duplicates first.
    """
    product_id = str(product_data["product_id"])
    
    # Check if product already exists
    vectorstore = create_vectorstore()
    existing = vectorstore.get(where={"product_id": product_id})
    
    if existing['ids']:
        logger.warning(f"Product {product_id} already exists. Use update_product_by_id instead")
        return False
    
    # Add new product
    try:
        document = _create_product_document(product_data)
        vectorstore.add_documents([document])
        vectorstore.persist()
        logger.info(f"Added new product: {product_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding product {product_id}: {e}")
        return False


def update_product_by_id(product_id: str) -> bool:
    """
    Update a specific product by product_id.
    Loads updated data from JSON file.
    """
    # Load products from JSON
    with open(PRODUCT_JSON_PATH, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # Find the product to update
    product_data = None
    for product in products:
        if str(product["product_id"]) == str(product_id):
            product_data = product
            break
    
    if not product_data:
        logger.error(f"Product {product_id} not found in JSON file")
        return False
    
    # Delete existing and add updated
    if delete_product_by_id(product_id):
        try:
            vectorstore = create_vectorstore()
            document = _create_product_document(product_data)
            vectorstore.add_documents([document])
            vectorstore.persist()
            logger.info(f"Updated product: {product_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating product {product_id}: {e}")
            return False
    
    return False


def update_all_products() -> None:
    """
    Update entire product collection by deleting all products and re-adding.
    """
    vectorstore = create_vectorstore()
    
    # Delete all products by source
    try:
        vectorstore.delete(where={"source": "product_json"})
        vectorstore.persist()
        logger.info("Deleted all existing products")
    except Exception as e:
        logger.warning(f"Error deleting existing products: {e}")
    
    # Re-add all products
    embed_products()


def embed_products_safe() -> None:
    """
    Embed products with duplicate checking - only adds new products.
    """
    # Load products from JSON
    with open(PRODUCT_JSON_PATH, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    vectorstore = create_vectorstore()
    new_documents = []
    skipped = 0
    
    for row in tqdm(products, desc="Checking for duplicates"):
        product_id = str(row["product_id"])
        
        # Check if product already exists
        existing = vectorstore.get(where={"product_id": product_id})
        
        if existing['ids']:
            skipped += 1
            continue
        
        # Add to new documents list
        document = _create_product_document(row)
        new_documents.append(document)
    
    # Add only new documents
    if new_documents:
        vectorstore.add_documents(new_documents)
        vectorstore.persist()
        logger.info(f"Added {len(new_documents)} new products")
    
    if skipped > 0:
        logger.info(f"Skipped {skipped} existing products")
    
    total_count = len(new_documents) + skipped
    logger.info(f"Total products processed: {total_count}")
