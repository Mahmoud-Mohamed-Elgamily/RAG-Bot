#!/usr/bin/env python3
"""
Example script demonstrating the optimized RAG search system.

This script shows how the hybrid search strategy improves both accuracy and performance
by using intelligent filtering, re-ranking, and diversity controls.
"""

import os
import time
from dotenv import load_dotenv
from src.rag_chatbot import RAGChatbot

# Load environment variables
load_dotenv()

def test_search_performance():
    """Test different performance modes and search strategies."""
    
    print("🚀 Testing Optimized RAG Search System")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Check if data is initialized
    if not chatbot.is_initialized():
        print("📊 Initializing data...")
        success = chatbot.initialize_data()
        if not success:
            print("❌ Failed to initialize data")
            return
        print("✅ Data initialized successfully")
    
    # Test queries with different complexity levels
    test_queries = [
        "3-bedroom villa in Dubai",  # Specific query with filters
        "apartments under 800000 AED",  # Price-based query
        "properties in Abu Dhabi",  # Location-based query
        "luxury penthouses",  # General semantic query
        "2 bedroom apartment in Dubai Marina with 2 bathrooms"  # Complex multi-filter query
    ]
    
    # Test different performance modes
    modes = ['fast', 'balanced', 'accurate']
    
    for mode in modes:
        print(f"\n🔧 Testing {mode.upper()} mode:")
        print("-" * 30)
        
        chatbot.set_performance_mode(mode)
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            start_time = time.time()
            result = chatbot.chat(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            properties_found = result.get('properties_found', 0)
            
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"🏠 Properties found: {properties_found}")
            print(f"💬 Response preview: {result['response'][:100]}...")
            
            # Show performance metrics
            if properties_found > 0:
                print(f"📊 Efficiency: {properties_found/response_time:.1f} properties/second")

def demonstrate_hybrid_search():
    """Demonstrate the hybrid search features."""
    
    print("\n🔍 Demonstrating Hybrid Search Features")
    print("=" * 50)
    
    chatbot = RAGChatbot()
    
    if not chatbot.is_initialized():
        print("❌ Data not initialized. Please run initialization first.")
        return
    
    # Example conversation to show follow-up handling
    conversation_history = []
    
    queries = [
        "Show me 3-bedroom apartments in Dubai",
        "What about the prices of these properties?",  # Follow-up query
        "Any villas in the same area?",  # Context-aware follow-up
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🗣️  Query {i}: '{query}'")
        
        start_time = time.time()
        result = chatbot.chat(query, conversation_history)
        end_time = time.time()
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": result['response']})
        
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        print(f"🏠 Properties found: {result.get('properties_found', 0)}")
        print(f"💬 Response: {result['response']}")
        
        if i < len(queries):
            print("\n" + "─" * 40)

def show_configuration_options():
    """Show available configuration options."""
    
    print("\n⚙️  Configuration Options")
    print("=" * 50)
    
    config_options = {
        'MAX_SEARCH_RESULTS': '20 (Max results from vector search)',
        'MAX_FINAL_RESULTS': '15 (Max results sent to LLM)',
        'ENABLE_DIVERSITY_FILTER': 'true (Enable diversity filtering)',
        'ENABLE_RERANKING': 'true (Enable result re-ranking)',
        'MAX_RESULTS': '10 (Max results returned to user)',
        'SIMILARITY_THRESHOLD': '0.3 (Minimum similarity score)'
    }
    
    print("Environment variables for tuning performance:")
    for key, description in config_options.items():
        current_value = os.getenv(key, 'default')
        print(f"  {key}={current_value}")
        print(f"    {description}")
        print()
    
    print("Performance Modes (set via code):")
    print("  chatbot.set_performance_mode('fast')    # 8-10 results, no diversity/reranking")
    print("  chatbot.set_performance_mode('balanced') # 15-20 results, full features")
    print("  chatbot.set_performance_mode('accurate') # 20-30 results, maximum accuracy")

if __name__ == "__main__":
    try:
        # Show configuration options
        show_configuration_options()
        
        # Test search performance
        test_search_performance()
        
        # Demonstrate hybrid search
        demonstrate_hybrid_search()
        
        print("\n✅ Testing completed!")
        print("\n💡 Key Improvements:")
        print("   • Reduced result set size (15-20 vs 30+)")
        print("   • Intelligent pre-filtering and re-ranking")
        print("   • Context-aware follow-up handling")
        print("   • Configurable performance modes")
        print("   • Diversity filtering for better variety")
        print("   • Faster LLM processing with optimized context")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Make sure you have:")
        print("  1. Set up your .env file with API keys")
        print("  2. Installed all dependencies")
        print("  3. Initialized the data") 