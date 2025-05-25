"""
UAE Real Estate RAG Chatbot

A Retrieval-Augmented Generation system for searching UAE real estate properties.
"""

__version__ = "1.0.0"
__author__ = "UAE Real Estate RAG Team"
__email__ = "contact@example.com"

from .data_processor import DataProcessor
from .vector_store import VectorStore
from .rag_chatbot import RAGChatbot

__all__ = ["DataProcessor", "VectorStore", "RAGChatbot"] 