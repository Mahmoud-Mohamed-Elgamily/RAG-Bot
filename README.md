# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables users to search for UAE real estate properties based on specific attributes such as bedrooms, bathrooms, location, and price. The system uses vector similarity search and OpenAI's language models to provide natural, conversational responses about available properties.

![UAE Real Estate RAG Chatbot](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)

## 🌟 Features

- **Natural Language Search**: Query properties using conversational language
- **Attribute-based Filtering**: Search by bedrooms, bathrooms, location, price range, and property type
- **Hybrid Search Strategy**: Optimized retrieval combining filtering, semantic search, and re-ranking
- **Performance Modes**: Configurable speed vs accuracy trade-offs (Fast/Balanced/Accurate)
- **Intelligent Context Management**: Reduced result sets with higher relevance for faster LLM processing
- **Vector Similarity Search**: Powered by ChromaDB and sentence transformers
- **Interactive Chat Interface**: User-friendly Streamlit web application
- **Real-time Analytics**: Visual charts showing price distribution and location breakdown
- **Flexible Data Source**: Supports Kaggle dataset or sample data generation
- **OpenAI Integration**: Supports both OpenAI and Azure OpenAI APIs

## 🏗️ Architecture

```
├── src/
│   ├── data_processor.py      # Data loading, cleaning, and preparation
│   ├── vector_store.py        # ChromaDB vector database operations
│   └── rag_chatbot.py         # Main RAG chatbot logic
├── streamlit_app.py           # Streamlit web interface
├── setup.py                   # Automated setup script
├── requirements.txt           # Python dependencies
├── env_example.txt           # Environment variables template
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI or OpenAI credentials
- Kaggle credentials for dataset download

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/uae-real-estate-rag.git
cd uae-real-estate-rag
```

2. **Run the automated setup**
```bash
# Edit the .env file created by setup with your API keys
nano .env
```

3. **Configure environment variables**
```bash
python3 setup.py
```

4. **Run the application**
```bash
python3 -m streamlit run streamlit_app.py
```

5. **Initialize the system**
   - Open the app in your browser (usually http://localhost:8501)
   - Click "🚀 Initialize System" in the sidebar
   - Wait for data processing and indexing to complete

### Manual Installation (Alternative)

If you prefer manual setup:

```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your credentials

# Install dependencies
pip install -r requirements.txt

# Then run the application
python3 -m streamlit run streamlit_app.py
```

## 📊 Example Queries

The chatbot can handle various types of real estate queries:

- **Basic searches**: "Find me a 3-bedroom villa in Abu Dhabi"
- **Price filtering**: "Show apartments in Dubai with 2 bathrooms under 800,000 AED"
- **Location specific**: "Properties in Sharjah with 4 bedrooms"
- **Property type**: "Penthouses in Dubai Marina"
- **Budget searches**: "Studios under 500,000 AED"

### Kaggle Dataset
The system is designed to work with the [UAE Real Estate 2024 Dataset](https://www.kaggle.com/datasets/mahmoudharidy/uae-real-estate-2024) from Kaggle.

### Sample Data
If the Kaggle dataset is not available, the system automatically generates sample data with:
- 5000 synthetic properties
- Realistic UAE locations (Dubai, Abu Dhabi, Sharjah, etc.)
- Various property types (Apartment, Villa, Townhouse, etc.)
- Diverse pricing and specifications

### Data Processor (`src/data_processor.py`)
- Downloads and cleans real estate data
- Handles missing values and data standardization
- Prepares documents for vector indexing
- Generates sample data when needed

### Vector Store (`src/vector_store.py`)
- Manages ChromaDB vector database
- Generates embeddings using sentence transformers
- Performs similarity search with filtering
- Handles metadata-based queries

### RAG Chatbot (`src/rag_chatbot.py`)
- Processes natural language queries
- Extracts search filters from text
- Integrates vector search with LLM generation
- Supports both OpenAI and Azure OpenAI

### Streamlit Interface (`streamlit_app.py`)
- Provides web-based chat interface
- Displays search results and analytics
- Manages system initialization and status
- Includes example queries and help

### Natural Language Processing
The system can extract various attributes from natural language:
- **Bedrooms**: "3-bedroom", "3 bed", "3 br"
- **Bathrooms**: "2 bathrooms", "2 bath", "2 ba"
- **Price ranges**: "under 800,000 AED", "below 1M AED"
- **Locations**: "Dubai", "Abu Dhabi", "Dubai Marina"
- **Property types**: "villa", "apartment", "penthouse"

### Vector Search
- Uses sentence transformers for semantic search
- Combines text similarity with metadata filtering
- Configurable similarity thresholds
- Batch processing for efficient indexing

### Response Generation
- Context-aware responses using retrieved properties
- Fallback responses when LLM fails
- Structured output with property details
- Conversational and informative tone

## ⚡ Performance Optimization

The system includes advanced optimization features to balance accuracy and speed:

### Hybrid Search Strategy

The chatbot uses a multi-stage retrieval approach:

1. **Query Analysis**: Extracts explicit filters (bedrooms, price, location, etc.)
2. **Adaptive Search**: Adjusts search scope based on query specificity
3. **Smart Filtering**: Pre-filters results using database constraints
4. **Re-ranking**: Boosts relevance scores based on exact matches and keyword alignment
5. **Diversity Control**: Ensures variety in results across locations, types, and price ranges
6. **Context Optimization**: Provides enriched context with relevance indicators

### Performance Modes

Configure the system for different use cases:

```python
from src.rag_chatbot import RAGChatbot

chatbot = RAGChatbot()

# Fast mode: 8-10 results, optimized for speed
chatbot.set_performance_mode('fast')

# Balanced mode: 15-20 results, good speed/accuracy balance (default)
chatbot.set_performance_mode('balanced')

# Accurate mode: 20-30 results, maximum accuracy
chatbot.set_performance_mode('accurate')
```

### Configuration Options

Fine-tune performance via environment variables:

```bash
# Search result limits
MAX_SEARCH_RESULTS=20          # Max results from vector search
MAX_FINAL_RESULTS=15           # Max results sent to LLM
MAX_RESULTS=10                 # Max results returned to user

# Feature toggles
ENABLE_DIVERSITY_FILTER=true   # Enable diversity filtering
ENABLE_RERANKING=true          # Enable result re-ranking

# Search parameters
SIMILARITY_THRESHOLD=0.3       # Minimum similarity score
```

### Performance Improvements

Compared to the basic approach:

- **Reduced Context Size**: 15-20 results vs 30+ (33-50% reduction)
- **Faster LLM Processing**: Optimized context with relevance indicators
- **Better Accuracy**: Pre-filtering and re-ranking improve result quality
- **Intelligent Follow-ups**: Context-aware handling of conversation flow
- **Adaptive Search**: Query complexity determines search strategy

### Common Issues

1. **"OpenAI API Key not found"**
   - Ensure you've created a `.env` file with your API key
   - Check that the key is valid and has sufficient credits

2. **"Failed to download dataset"**
   - Verify Kaggle credentials in `.env` file
   - The system will use sample data if Kaggle fails

3. **"No properties found"**
   - Try more general queries
   - Check if data was properly indexed
   - Lower the similarity threshold in `.env`

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)
