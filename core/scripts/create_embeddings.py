#!/usr/bin/env python3
"""
Complete Embedding Script
Creates embeddings for both products and documents in ChromaDB.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.embed_products import embed_products, embed_products_safe
from core.embed_documents import embed_documents, list_document_sources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_all_embeddings(safe_mode: bool = False):
    """
    Create embeddings for both products and documents.
    
    Args:
        safe_mode: If True, uses safe embedding (skips duplicates)
    """
    logger.info("Starting complete embedding process")
    
    try:
        # Embed products
        logger.info("Step 1: Embedding products")
        if safe_mode:
            embed_products_safe()
        else:
            embed_products()
        
        # Embed documents
        logger.info("Step 2: Embedding documents")
        
        # List available document sources
        sources = list_document_sources()
        if sources:
            logger.info(f"Found document sources: {sources}")
            embed_documents()  # Process all .txt files
        else:
            logger.warning("No document sources found")
        
        logger.info("Complete embedding process finished successfully")
        
    except Exception as e:
        logger.error(f"Error during embedding process: {e}")
        raise


def main():
    """
    Main function with command line argument handling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create embeddings for products and documents"
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Use safe mode (skip duplicate products)"
    )
    parser.add_argument(
        "--products-only",
        action="store_true",
        help="Only embed products, skip documents"
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Only embed documents, skip products"
    )
    
    args = parser.parse_args()
    
    try:
        if args.products_only:
            logger.info("Embedding products only")
            if args.safe:
                embed_products_safe()
            else:
                embed_products()
        
        elif args.documents_only:
            logger.info("Embedding documents only")
            embed_documents()
        
        else:
            logger.info("Embedding both products and documents")
            create_all_embeddings(safe_mode=args.safe)
        
        logger.info("Embedding script completed successfully")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
