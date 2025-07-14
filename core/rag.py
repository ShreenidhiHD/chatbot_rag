"""
RAG (Retrieval-Augmented Generation) System
Combines vector retrieval with LLM generation for contextual responses.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from .config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME
)
from .llm_service import call_llm, call_local_llm, call_openai_llm, call_gemini_llm
from .llm_utils import (
    classify_intent,
    get_chat_history_context,
    resolve_product_reference,
    format_rag_context,
    is_safe_query,
    get_specialized_prompt,
    get_agent_name,
    should_resolve_product_context,
    should_use_chat_history,
    format_sales_response
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Complete RAG system with retrieval and generation capabilities.
    """
    
    def __init__(self, llm_provider: str = "local"):
        self.llm_provider = llm_provider
        self.retriever = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the vector store retriever"""
        try:
            # Initialize embedding model
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
            # Load vector store
            vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_model,
                persist_directory=CHROMA_DB_DIR
            )
            
            # Create retriever
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Top 5 similar documents
            )
            
            logger.info("RAG retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document contents
        """
        try:
            # Update retriever with new k value
            self.retriever.search_kwargs = {"k": k}
            
            # Retrieve documents
            documents = self.retriever.get_relevant_documents(query)
            
            # Extract content from documents
            doc_contents = [doc.page_content for doc in documents]
            
            logger.info(f"Retrieved {len(doc_contents)} relevant documents")
            return doc_contents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline with intelligent context decisions.
        
        Args:
            query: User query
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Safety check
            if not is_safe_query(query):
                return {
                    "response": "I cannot provide information on harmful or dangerous topics.",
                    "intent": "blocked",
                    "agent": "Safety Filter",
                    "sources": [],
                    "products": [],
                    "metadata": {},
                    "error": "Unsafe query blocked"
                }
                
            # Initialize chat history if not provided
            if chat_history is None:
                chat_history = []
            
            # Step 1: Intent classification
            intent = classify_intent(query)
            
            # Step 2: Intelligent decision on whether to use chat history
            use_chat_history = should_use_chat_history(query, intent)
            
            # Step 3: Intelligent decision on whether to resolve product context
            use_product_resolution = should_resolve_product_context(query, intent)
            
            # Step 4: Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_documents(query)
            
            # Step 5: Get chat history context (only if needed)
            chat_context = ""
            if use_chat_history:
                chat_context = get_chat_history_context(chat_history)
            
            # Step 6: Resolve product references (only if needed)
            product_context = ""
            if use_product_resolution:
                product_context = resolve_product_reference(query, chat_history)
            
            # Step 7: Format context
            formatted_context = format_rag_context(
                retrieved_docs, 
                chat_context, 
                product_context
            )
            
            # Step 8: Get specialized prompt based on intent
            specialized_prompt = get_specialized_prompt(intent, formatted_context, query)
            
            # Step 9: Generate response using specialized agent
            response = self._call_llm_with_prompt(specialized_prompt)
            
            # Step 10: Format response with structured product information
            formatted_response = format_sales_response(response, intent)
            
            # Step 11: Get agent name
            agent_name = get_agent_name(intent)
            
            # Return structured response
            return {
                "response": formatted_response["text"],
                "intent": intent,
                "agent": agent_name,
                "sources": retrieved_docs,
                "context": formatted_context,
                "products": formatted_response["products"],
                "metadata": formatted_response["metadata"],
                "chat_history_used": use_chat_history,
                "product_context_used": use_product_resolution,
                "chat_context_actual": bool(chat_context),
                "product_context_actual": bool(product_context),
                "intelligent_decisions": True
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "intent": "error",
                "agent": "Error Handler",
                "sources": [],
                "error": str(e)
            }
    
    def _call_llm_with_prompt(self, prompt: str) -> str:
        """Call the configured LLM provider with custom prompt"""
        if self.llm_provider == "local":
            return call_local_llm("", "", custom_prompt=prompt)
        elif self.llm_provider == "openai":
            return call_openai_llm("", "", custom_prompt=prompt)
        elif self.llm_provider == "gemini":
            return call_gemini_llm("", "", custom_prompt=prompt)
        else:
            return call_llm("", "", provider=self.llm_provider, custom_prompt=prompt)
    
    def _call_llm(self, context: str, query: str) -> str:
        """Call the configured LLM provider"""
        if self.llm_provider == "local":
            return call_local_llm(context, query)
        elif self.llm_provider == "openai":
            return call_openai_llm(context, query)
        elif self.llm_provider == "gemini":
            return call_gemini_llm(context, query)
        else:
            return call_llm(context, query, provider=self.llm_provider)


# Convenience functions for direct usage
def create_rag_system(llm_provider: str = "local") -> RAGSystem:
    """Create and return a RAG system instance"""
    return RAGSystem(llm_provider=llm_provider)


def quick_rag_query(query: str, llm_provider: str = "local") -> str:
    """
    Quick RAG query without advanced features.
    
    Args:
        query: User query
        llm_provider: LLM provider to use
        
    Returns:
        Response string
    """
    rag_system = RAGSystem(llm_provider=llm_provider)
    result = rag_system.generate_response(query)
    return result["response"]


def advanced_rag_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_provider: str = "local"
) -> Dict[str, Any]:
    """
    Advanced RAG query with multi-agent system using intelligent context decisions.
    
    Args:
        query: User query
        chat_history: Previous conversation history
        llm_provider: LLM provider to use
        
    Returns:
        Complete response dictionary with agent information
    """
    rag_system = RAGSystem(llm_provider=llm_provider)
    return rag_system.generate_response(
        query,
        chat_history=chat_history
    )

