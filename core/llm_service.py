
import os
import logging
from typing import Optional
from .config import (
    LOCAL_LLM_MODEL_PATH,
    LOCAL_LLM_MODEL_NAME,
    DEFAULT_LLM_PROVIDER,
    MAX_TOKENS,
    TEMPERATURE,
    LLM_THREADS,
    LLM_CONTEXT_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """Unified LLM service supporting multiple providers with singleton pattern"""
    
    _instances = {}
    
    def __new__(cls, provider: str = DEFAULT_LLM_PROVIDER):
        if provider not in cls._instances:
            cls._instances[provider] = super().__new__(cls)
        return cls._instances[provider]
    
    def __init__(self, provider: str = DEFAULT_LLM_PROVIDER):
        if hasattr(self, 'initialized'):
            return
        self.provider = provider
        self.llm = None
        self._initialize_llm()
        self.initialized = True
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        if self.provider == "local":
            self._initialize_local_llm()
        elif self.provider == "openai":
            self._initialize_openai_llm()
        elif self.provider == "gemini":
            self._initialize_gemini_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _initialize_local_llm(self):
        """Initialize local LLM using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            model_path = os.path.expanduser(LOCAL_LLM_MODEL_PATH)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Local model not found: {model_path}")
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=LLM_CONTEXT_SIZE,
                n_threads=LLM_THREADS,
                verbose=False
            )
            logger.info(f"Initialized local LLM: {LOCAL_LLM_MODEL_NAME}")
            
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
    
    def _initialize_openai_llm(self):
        """Initialize OpenAI LLM using LangChain"""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                openai_api_key=api_key
            )
            logger.info("Initialized OpenAI LLM")
            
        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
    
    def _initialize_gemini_llm(self):
        """Initialize Gemini LLM using Google's Generative AI SDK"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Initialize the Gemini client
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel(model_name="gemini-2.0-flash-lite")
            logger.info("Initialized Gemini 2.0 Flash-Lite model")
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    def generate_response(self, context: str, query: str, custom_prompt: str = None) -> str:
        """Generate response using the configured LLM"""
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"""You are a safe and helpful AI assistant. 
            Never respond to questions that are violent, harmful, or illegal.

            Context:
            {context}

            Question: {query}

            Answer:"""
        
        if self.provider == "local":
            return self._generate_local_response(prompt)
        else:
            return self._generate_langchain_response(prompt)
    
    def _generate_local_response(self, prompt: str) -> str:
        """Generate response using local LLM"""
        try:
            response = self.llm(
                prompt,
                max_tokens=MAX_TOKENS,
                stop=["\nQ:", "\nQuestion:", "\nContext:"],
                temperature=TEMPERATURE
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            raise
    
    def _generate_langchain_response(self, prompt: str) -> str:
        """Generate response using LangChain or Gemini LLM"""
        try:
            if self.provider == "gemini":
                # Use direct Gemini API
                response = self.gemini_client.generate_content(prompt)
                return response.text.strip()
            else:
                # Use LangChain for other models
                from langchain.schema import HumanMessage
                
                message = HumanMessage(content=prompt)
                response = self.llm.invoke([message])
                return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


# Convenience functions for direct usage
def call_local_llm(context: str, query: str, custom_prompt: str = None) -> str:
    """Call local LLM directly"""
    service = LLMService(provider="local")
    return service.generate_response(context, query, custom_prompt)


def call_openai_llm(context: str, query: str, custom_prompt: str = None) -> str:
    """Call OpenAI LLM directly"""
    service = LLMService(provider="openai")
    return service.generate_response(context, query, custom_prompt)


def call_gemini_llm(context: str, query: str, custom_prompt: str = None) -> str:
    """Call Gemini LLM directly"""
    service = LLMService(provider="gemini")
    return service.generate_response(context, query, custom_prompt)


def call_llm(context: str, query: str, provider: str = DEFAULT_LLM_PROVIDER, custom_prompt: str = None) -> str:
    """Call any LLM provider"""
    service = LLMService(provider=provider)
    return service.generate_response(context, query, custom_prompt)

