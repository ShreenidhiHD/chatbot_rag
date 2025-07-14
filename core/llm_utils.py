"""
LLM Utilities Module
Contains utility functions for LLM operations like intent classification, etc.
"""

import logging
import re
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_intent(query: str) -> str:
    """
    Classify the intent of user query.
    
    Args:
        query: User input query
        
    Returns:
        Intent classification (sales, refund, general, etc.)
    """
    query_lower = query.lower()
    
    # Sales intent keywords
    sales_keywords = [
        "buy", "purchase", "price", "cost", "order", "product", "coffee", "beans",
        "available", "stock", "catalog", "shop", "store", "wholesale", "retail",
        "discount", "offer", "promo", "new", "recommendation", "suggest"
    ]
    
    # Refund intent keywords
    refund_keywords = [
        "refund", "return", "exchange", "cancel", "money back", "replacement",
        "damaged", "defective", "wrong", "mistake", "complaint", "issue"
    ]
    
    # Support intent keywords
    support_keywords = [
        "help", "support", "contact", "hours", "location", "store", "delivery",
        "shipping", "payment", "account", "login", "register"
    ]
    
    # Check for sales intent
    if any(keyword in query_lower for keyword in sales_keywords):
        logger.info(f"Classified as SALES intent: {query[:50]}...")
        return "sales"
    
    # Check for refund intent
    if any(keyword in query_lower for keyword in refund_keywords):
        logger.info(f"Classified as REFUND intent: {query[:50]}...")
        return "refund"
    
    # Check for support intent
    if any(keyword in query_lower for keyword in support_keywords):
        logger.info(f"Classified as SUPPORT intent: {query[:50]}...")
        return "support"
    
    # Default to general
    logger.info(f"Classified as GENERAL intent: {query[:50]}...")
    return "general"


def get_chat_history_context(chat_history: List[Dict[str, str]], limit: int = 5) -> str:
    """
    Get formatted chat history context from last N messages.
    
    Args:
        chat_history: List of chat messages with 'user' and 'assistant' keys
        limit: Number of last messages to include
        
    Returns:
        Formatted context string
    """
    # TODO: Implement proper chat history retrieval from database
    # For now, return empty string
    logger.info(f"Getting chat history context (limit: {limit})")
    return ""


def resolve_product_reference(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Resolve product references in query using chat history context.
    
    Args:
        query: User input query
        chat_history: Previous conversation context
        
    Returns:
        Additional product context (for now returns empty string)
    """
    # TODO: Implement product reference resolution
    # This would analyze chat history to find product mentions
    # and return relevant product information
    logger.info(f"Resolving product references for query: {query[:50]}...")
    return ""


def format_rag_context(retrieved_docs: List[str], chat_context: str, product_context: str) -> str:
    """
    Format all context components into a single context string for LLM.
    
    Args:
        retrieved_docs: Documents retrieved from vector store
        chat_context: Previous conversation context
        product_context: Product-specific context
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # Add retrieved documents
    if retrieved_docs:
        context_parts.append("Retrieved Information:")
        context_parts.extend(retrieved_docs)
    
    # Add chat history context
    if chat_context:
        context_parts.append("\nPrevious Conversation:")
        context_parts.append(chat_context)
    
    # Add product context
    if product_context:
        context_parts.append("\nProduct Information:")
        context_parts.append(product_context)
    
    return "\n".join(context_parts)


def is_safe_query(query: str) -> bool:
    """
    Check if query is safe (not harmful/violent/illegal).
    
    Args:
        query: User input query
        
    Returns:
        True if query is safe, False otherwise
    """
    banned_words = ["kill", "murder", "harm", "die", "bomb", "weapon", "stab", "suicide"]
    
    query_lower = query.lower()
    for word in banned_words:
        if word in query_lower:
            logger.warning(f"Unsafe query detected: contains '{word}'")
            return False
    
    return True


def get_specialized_prompt(intent: str, context: str, query: str) -> str:
    """
    Get specialized prompt based on intent.
    
    Args:
        intent: Classified intent (sales, refund, support, general)
        context: Retrieved context information
        query: User query
        
    Returns:
        Specialized prompt for the intent
    """
    base_safety = "You are a helpful AI assistant. Never respond to questions that are violent, harmful, or illegal."
    
    if intent == "sales":
        return f"""{base_safety}

You are a coffee sales specialist. Your goal is to help customers find the perfect coffee products and make purchases.

Key Guidelines:
- Be enthusiastic about coffee products
- Highlight product benefits and features
- Suggest complementary products
- Mention pricing and availability
- Guide towards making a purchase
- Ask clarifying questions about preferences
- For specific products, always include product ID, name, and price
- Format product information clearly for easy UI integration

Response Format:
When mentioning specific products, use this format:
**Product Name** (ID: product_id) - $price
- Product description/features
- [Available in store/online]

Context:
{context}

Customer Question: {query}

Sales Response:"""

    elif intent == "refund":
        return f"""{base_safety}

You are a customer service specialist handling refunds and returns.

Key Guidelines:
- Be empathetic and understanding
- Clearly explain refund policies
- Provide step-by-step instructions
- Mention timelines and requirements
- Offer alternative solutions
- Be professional and helpful

Context:
{context}

Customer Question: {query}

Customer Service Response:"""

    elif intent == "support":
        return f"""{base_safety}

You are a customer support specialist providing general assistance.

Key Guidelines:
- Be helpful and informative
- Provide accurate store information
- Explain processes clearly
- Offer multiple contact options
- Be patient and thorough
- Direct to appropriate resources

Context:
{context}

Customer Question: {query}

Support Response:"""

    else:  # general
        return f"""{base_safety}

You are a knowledgeable coffee store assistant providing general information.

Key Guidelines:
- Be friendly and informative
- Provide accurate information
- Be concise but complete
- Offer to help further
- Stay within your knowledge

Context:
{context}

Question: {query}

Response:"""


def get_agent_name(intent: str) -> str:
    """
    Get agent name based on intent.
    
    Args:
        intent: Classified intent
        
    Returns:
        Agent name string
    """
    agent_names = {
        "sales": "Sales Specialist",
        "refund": "Customer Service Agent",
        "support": "Support Agent",
        "general": "Coffee Assistant"
    }
    
    return agent_names.get(intent, "Assistant")


def should_resolve_product_context(query: str, intent: str) -> bool:
    """
    Determine if product context resolution is needed based on query and intent.
    
    Args:
        query: User query
        intent: Classified intent
        
    Returns:
        True if product context should be resolved, False otherwise
    """
    query_lower = query.lower()
    
    # Keywords that suggest need for specific product information
    product_context_keywords = [
        "this", "that", "it", "the one", "same", "different", "another",
        "previous", "last", "earlier", "mentioned", "discussed",
        "compare", "vs", "versus", "difference between",
        "similar", "like that", "alternative"
    ]
    
    # Reference words that suggest continuing previous conversation
    reference_keywords = [
        "this product", "that coffee", "the beans", "same order",
        "my order", "my coffee", "my purchase", "what I bought"
    ]
    
    # Always check for product context in sales intent with references
    if intent == "sales":
        if any(keyword in query_lower for keyword in product_context_keywords + reference_keywords):
            logger.info(f"Product context needed for sales query: {query[:50]}...")
            return True
    
    # Check for refund/exchange scenarios
    if intent == "refund":
        if any(keyword in query_lower for keyword in reference_keywords):
            logger.info(f"Product context needed for refund query: {query[:50]}...")
            return True
    
    # Check for comparison or follow-up questions
    if any(keyword in query_lower for keyword in product_context_keywords):
        logger.info(f"Product context needed for reference query: {query[:50]}...")
        return True
    
    logger.info(f"No product context needed for query: {query[:50]}...")
    return False


def should_use_chat_history(query: str, intent: str) -> bool:
    """
    Determine if chat history context is needed based on query and intent.
    
    Args:
        query: User query
        intent: Classified intent
        
    Returns:
        True if chat history should be used, False otherwise
    """
    query_lower = query.lower()
    
    # Keywords that suggest need for conversation context
    context_keywords = [
        "continue", "also", "and", "what about", "how about",
        "yes", "no", "okay", "sure", "thanks", "thank you",
        "previous", "earlier", "before", "last time",
        "again", "still", "more", "else", "other"
    ]
    
    # Short queries often need context
    if len(query.split()) <= 3:
        logger.info(f"Chat history needed for short query: {query}")
        return True
    
    # Questions with context references
    if any(keyword in query_lower for keyword in context_keywords):
        logger.info(f"Chat history needed for contextual query: {query[:50]}...")
        return True
    
    # Always use history for follow-up refund questions
    if intent == "refund":
        logger.info(f"Chat history needed for refund query: {query[:50]}...")
        return True
    
    logger.info(f"No chat history needed for query: {query[:50]}...")
    return False


def extract_product_info(response: str) -> Dict[str, Any]:
    """
    Extract structured product information from sales response.
    
    Args:
        response: Sales agent response text
        
    Returns:
        Dictionary containing products mentioned and metadata
    """
    # Pattern to match product format: **Product Name** (ID: product_id) - $price
    product_pattern = r'\*\*([^*]+)\*\*\s*\(ID:\s*([^)]+)\)\s*-\s*\$([0-9.]+)'
    
    products = []
    matches = re.findall(product_pattern, response)
    
    for match in matches:
        product_name, product_id, price = match
        products.append({
            "id": product_id.strip(),
            "name": product_name.strip(),
            "price": float(price.strip()),
            "buy_link": f"/product/{product_id.strip()}",
            "image_url": f"/images/{product_id.strip()}.jpg"
        })
    
    return {
        "products": products,
        "total_products": len(products),
        "response_type": "sales",
        "has_products": len(products) > 0
    }


def format_sales_response(response: str, intent: str) -> Dict[str, Any]:
    """
    Format response with structured product information for UI integration.
    
    Args:
        response: Raw LLM response
        intent: Intent classification
        
    Returns:
        Formatted response with product info
    """
    result = {
        "text": response,
        "intent": intent,
        "agent": get_agent_name(intent),
        "products": [],
        "metadata": {}
    }
    
    if intent == "sales":
        product_info = extract_product_info(response)
        result["products"] = product_info["products"]
        result["metadata"] = {
            "total_products": product_info["total_products"],
            "has_products": product_info["has_products"]
        }
    
    return result
