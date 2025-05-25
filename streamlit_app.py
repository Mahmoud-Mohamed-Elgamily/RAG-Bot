import streamlit as st
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Import our modules
try:
    from src.rag_chatbot import RAGChatbot
except ImportError:
    st.error("Error importing RAG chatbot modules. Please ensure all dependencies are installed.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="UAE Real Estate RAG Chatbot",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>

    .main {
        max-height: 86vh;
        overflow-y: auto;
    }

    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
        background-color: #f8f9fa;
    }
    
    .chat-message h2,
    .chat-message h3,
    .chat-message h4,
    .chat-message h5,
    .chat-message h6 {
        color: #475c92;
    }

    .chat-message hr {
        background-color: #475c92;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #ddd;
        margin: 0.5rem;
    }
    
    .status-operational {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .block-container {
        padding: 2rem 3rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_chatbot():
    """Initialize the RAG chatbot with caching."""
    try:
        chatbot = RAGChatbot()
        return chatbot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None

def main():    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Auto-initialize if not already done
    if not st.session_state.initialized and not st.session_state.chatbot:
        with st.spinner("Loading system..."):
            chatbot = init_chatbot()
            if chatbot:
                st.session_state.chatbot = chatbot
                # Check if data already exists
                if chatbot.is_initialized():
                    st.session_state.initialized = True
                else:
                    st.info("👈 Click 'Reload Property Data' in the sidebar to load property data.")
    
    # Sidebar
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏠 UAE Real Estate RAG Chatbot</h1>
            <p>Find your perfect property in the UAE with AI-powered search</p>
        </div>
        """, unsafe_allow_html=True)
        st.header("🔧 System Control")
        
        # Environment check
        st.subheader("Environment Status")
        openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        if openai_key:
            st.success("✅ AZURE OPENAI API Key configured")
        else:
            st.error("❌ AZURE OPENAI API KEY not found")
            st.warning("Please set your AZURE_OPENAI_API_KEY in the .env file")
        
        # Show current status
        if st.session_state.chatbot and st.session_state.initialized:
            st.success("✅ System Ready - No initialization needed!")
        
        # Initialize/Re-initialize data button
        button_text = "🔄 Reload Property Data" if st.session_state.initialized else "🚀 Initialize System"
        if st.button(button_text, type="primary"):
            if not openai_key:
                st.error("Please configure OpenAI API key first")
            else:
                with st.spinner("Initializing RAG system..."):
                    if not st.session_state.chatbot:
                        chatbot = init_chatbot()
                        if chatbot:
                            st.session_state.chatbot = chatbot
                    
                    if st.session_state.chatbot:
                        # Force re-index if this is a re-initialization
                        force_reindex = st.session_state.initialized
                        success = st.session_state.chatbot.initialize_data(force_reindex=force_reindex)
                        if success:
                            st.session_state.initialized = True
                            st.success("✅ System initialized successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize data")
                    else:
                        st.error("❌ Failed to initialize chatbot")
        
        # System status
        if st.session_state.chatbot:
            st.subheader("System Status")
            stats = st.session_state.chatbot.get_system_stats()
            
            if stats.get('status') == 'operational':
                st.markdown('<p class="status-operational">🟢 Operational</p>', unsafe_allow_html=True)
                st.metric("Properties Indexed", stats['vector_store']['total_documents'])
                st.metric("Model", stats['model'])
            else:
                st.markdown('<p class="status-error">🔴 Error</p>', unsafe_allow_html=True)
                st.error(f"Error: {stats.get('error', 'Unknown error')}")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Find me a 3-bedroom villa in Abu Dhabi",
            "Show apartments in Dubai with 2 bathrooms under 800,000 AED",
            "Properties in Sharjah with 4 bedrooms",
            "Penthouses in Dubai Marina",
            "Studios under 500,000 AED"
        ]
        
        for query in example_queries:
            if st.button(f"'{query}'", key=f"example_{hash(query)}"):
                if st.session_state.initialized:
                    # Add to chat and process
                    user_message = {"role": "user", "content": query, "timestamp": datetime.now()}
                    st.session_state.chat_history.append(user_message)
                    
                    # Process with chatbot (with conversation history)
                    with st.spinner("Searching properties..."):
                        response_data = st.session_state.chatbot.chat(query, st.session_state.chat_history)
                        
                        bot_message = {
                            "role": "assistant", 
                            "content": response_data['response'],
                            "timestamp": datetime.now()
                        }
                        st.session_state.chat_history.append(bot_message)
                    st.rerun()
                else:
                    st.warning("Please initialize the system first")
    
    # Main content area
    st.header("💬 Chat Interface")
    
    # Chat interface
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system using the sidebar to start chatting!")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message" style="background-color: #e3f2fd; color: black;">
                        <strong>You:</strong> {message["content"]}
                        <br><small>{message["timestamp"].strftime("%H:%M:%S")}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message" style="background-color: #f1f8e9; color: black;">
                        <strong>Assistant:</strong> {message["content"]}
                        <br><small>{message["timestamp"].strftime("%H:%M:%S")}</small>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat input - MUST be at main level, outside all containers
    if st.session_state.initialized:
        user_input = st.chat_input(
            placeholder="e.g., Find me a 2-bedroom apartment in Dubai under 1,000,000 AED",
            key="chat_input"
        )
        
        # Process chat input
        if user_input and user_input.strip():
            # Add user message
            user_message = {"role": "user", "content": user_input, "timestamp": datetime.now()}
            st.session_state.chat_history.append(user_message)
            
            # Process with chatbot (with conversation history)
            with st.spinner("Searching properties..."):
                response_data = st.session_state.chatbot.chat(user_input, st.session_state.chat_history)
                
                bot_message = {
                    "role": "assistant", 
                    "content": response_data['response'],
                    "timestamp": datetime.now()
                }
                st.session_state.chat_history.append(bot_message)
            
            st.rerun()

if __name__ == "__main__":
    main() 