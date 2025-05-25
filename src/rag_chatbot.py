import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
import re
from datetime import datetime
from .vector_store import VectorStore
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """RAG-based chatbot for UAE real estate property search."""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = self._init_openai_client()
        
        # Initialize components
        self.vector_store = VectorStore(
            persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', 'chroma_db'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )
        self.data_processor = DataProcessor()
        
        # Configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if azure_endpoint:
            # For Azure OpenAI, use deployment name
            self.model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
        else:
            # For standard OpenAI, use model name
            self.model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-2024-08-06')
        
        self.max_results = int(os.getenv('MAX_RESULTS', 10))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
        
        # Hybrid search configuration
        self.max_search_results = int(os.getenv('MAX_SEARCH_RESULTS', 20))  # Max results from vector search
        self.max_final_results = int(os.getenv('MAX_FINAL_RESULTS', 15))    # Max results sent to LLM
        self.enable_diversity_filter = os.getenv('ENABLE_DIVERSITY_FILTER', 'true').lower() == 'true'
        self.enable_reranking = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
        
        logger.info("RAG Chatbot initialized successfully")
    
    def _init_openai_client(self) -> OpenAI:
        """Initialize OpenAI client with proper configuration."""
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        
        # Check if using Azure OpenAI
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if azure_endpoint:
            # Azure OpenAI configuration
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15'),
                azure_endpoint=azure_endpoint
            )
        else:
            # Standard OpenAI configuration
            base_url = os.getenv('OPENAI_BASE_URL')
            if base_url:
                # Custom OpenAI endpoint (e.g., proxy or custom deployment)
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                # Standard OpenAI API
                return OpenAI(api_key=api_key)
    
    def initialize_data(self, force_reindex: bool = False) -> bool:
        """Load and index the real estate data."""
        try:
            logger.info("Checking data initialization status...")
            
            # Check if data is already indexed
            stats = self.vector_store.get_collection_stats()
            existing_count = stats.get('total_documents', 0)
            
            if existing_count > 0 and not force_reindex:
                logger.info(f"Data already initialized with {existing_count} properties. Skipping re-indexing.")
                return True
            
            logger.info("Loading and processing real estate data...")
            
            # Load and clean data
            df = self.data_processor.load_and_clean_data()
            if df.empty:
                logger.error("No data loaded")
                return False
            
            # Prepare documents for indexing
            documents = self.data_processor.prepare_documents_for_indexing(df)
            
            # Index documents in vector store
            success = self.vector_store.index_documents(documents, force_reindex=force_reindex)
            
            if success:
                stats = self.vector_store.get_collection_stats()
                logger.info(f"Data initialization complete. Indexed {stats['total_documents']} properties")
                return True
            else:
                logger.error("Failed to index documents")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing data: {str(e)}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if the system is already initialized with data."""
        try:
            stats = self.vector_store.get_collection_stats()
            return stats.get('total_documents', 0) > 0
        except Exception as e:
            logger.error(f"Error checking initialization status: {str(e)}")
            return False
    
    def chat(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process user query and return response with property search results."""
        try:
            # Simple classification: is this about properties or general chat?
            if self._is_property_related(user_query, conversation_history):
                # Use hybrid retrieval strategy for better accuracy and performance
                search_results = self._hybrid_search(user_query, conversation_history)
                
                logger.info(f"Hybrid search returned {len(search_results)} results")
                
                if not search_results:
                    return {
                        'response': "I couldn't find any properties in the database. Please try a different search or check if the system is properly initialized.",
                        'properties_found': 0,
                        'search_filters': {},
                        'properties': []
                    }
                
                # Generate response with optimized context
                response = self._generate_smart_response(user_query, search_results, conversation_history)
                
                return {
                    'response': response,
                    'properties_found': len(search_results),
                    'search_filters': {},
                    'properties': [result['metadata'] for result in search_results[:self.max_results]]
                }
            else:
                # Handle general conversation
                response = self._handle_general_conversation(user_query, conversation_history)
                return {
                    'response': response,
                    'properties_found': 0,
                    'search_filters': {},
                    'properties': []
                }
            
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'properties_found': 0,
                'search_filters': {},
                'properties': []
            }

    def _hybrid_search(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Hybrid search strategy combining filtering, semantic search, and ranking."""
        try:
            # Step 1: Extract explicit filters from query
            extracted_filters = self._extract_filters_from_query(user_query)
            logger.info(f"Extracted filters: {extracted_filters}")
            
            # Step 2: Check if this is a follow-up query that might reference previous results
            is_followup = self._is_followup_query(user_query, conversation_history)
            
            if is_followup and conversation_history:
                # For follow-ups, use a smaller, more focused search
                search_results = self.vector_store.search(
                    query=user_query,
                    n_results=min(self.max_final_results, self.max_results * 2),  # Use config
                    filters=extracted_filters if extracted_filters else None
                )
            else:
                # Step 3: Determine search strategy based on query specificity
                if extracted_filters:
                    # Specific query with filters - use database filtering first
                    search_results = self.vector_store.search(
                        query=user_query,
                        n_results=self.max_search_results,  # Use config
                        filters=extracted_filters
                    )
                    
                    # If filtered search returns too few results, expand search
                    if len(search_results) < 5:
                        logger.info("Filtered search returned few results, expanding search...")
                        expanded_results = self.vector_store.search(
                            query=user_query,
                            n_results=min(self.max_search_results + 5, 25),  # Slightly expand
                            filters=None  # Remove filters for broader search
                        )
                        # Combine and deduplicate
                        search_results = self._merge_and_deduplicate_results(search_results, expanded_results)
                else:
                    # General query - use semantic search with moderate scope
                    search_results = self.vector_store.search(
                        query=user_query,
                        n_results=self.max_search_results,  # Use config
                        filters=None
                    )
            
            # Step 4: Apply post-processing filters and ranking
            if extracted_filters and not is_followup:
                # Apply additional filtering that wasn't handled by vector store
                search_results = self._apply_filters(search_results, extracted_filters)
            
            # Step 5: Re-rank results based on query relevance and diversity (if enabled)
            if self.enable_reranking:
                search_results = self._rerank_results(search_results, user_query, extracted_filters)
            
            # Step 6: Apply diversity filtering (if enabled)
            if self.enable_diversity_filter and len(search_results) > 10:
                search_results = self._apply_diversity_filter(search_results)
            
            # Step 7: Limit final results to manageable size
            final_results = search_results[:self.max_final_results]
            
            logger.info(f"Final hybrid search results: {len(final_results)} properties")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to simple search
            return self.vector_store.search(
                query=user_query,
                n_results=self.max_results,
                filters=None
            )

    def _merge_and_deduplicate_results(self, primary_results: List[Dict[str, Any]], secondary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge two result sets and remove duplicates."""
        seen_ids = set()
        merged_results = []
        
        # Add primary results first (higher priority)
        for result in primary_results:
            prop_id = result['metadata'].get('title', '') + str(result['metadata'].get('price', 0))
            if prop_id not in seen_ids:
                seen_ids.add(prop_id)
                merged_results.append(result)
        
        # Add secondary results if not already present
        for result in secondary_results:
            prop_id = result['metadata'].get('title', '') + str(result['metadata'].get('price', 0))
            if prop_id not in seen_ids:
                seen_ids.add(prop_id)
                merged_results.append(result)
        
        return merged_results

    def _rerank_results(self, search_results: List[Dict[str, Any]], user_query: str, extracted_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-rank search results based on relevance and diversity."""
        if not search_results:
            return search_results
        
        query_lower = user_query.lower()
        
        # Calculate relevance scores
        for result in search_results:
            metadata = result['metadata']
            relevance_score = result.get('similarity_score', 0.5)
            
            # Boost score for exact filter matches
            if extracted_filters:
                if 'property_type' in extracted_filters:
                    prop_type = metadata.get('property_type', '').lower()
                    filter_type = extracted_filters['property_type'].lower()
                    if filter_type in prop_type:
                        relevance_score += 0.2
                
                if 'bedrooms' in extracted_filters:
                    if metadata.get('bedrooms', 0) == extracted_filters['bedrooms']:
                        relevance_score += 0.15
                
                if 'location' in extracted_filters:
                    prop_location = metadata.get('location', '').lower()
                    filter_location = extracted_filters['location'].lower()
                    if filter_location in prop_location:
                        relevance_score += 0.1
            
            # Boost for keyword matches in query
            title = metadata.get('title', '').lower()
            location = metadata.get('location', '').lower()
            prop_type = metadata.get('property_type', '').lower()
            
            # Check for direct keyword matches
            if any(word in title for word in query_lower.split()):
                relevance_score += 0.1
            if any(word in location for word in query_lower.split()):
                relevance_score += 0.1
            if any(word in prop_type for word in query_lower.split()):
                relevance_score += 0.1
            
            result['final_relevance_score'] = min(relevance_score, 1.0)  # Cap at 1.0
        
        # Sort by final relevance score
        search_results.sort(key=lambda x: x.get('final_relevance_score', 0), reverse=True)
        
        # Apply diversity filtering to avoid too many similar properties
        diverse_results = self._apply_diversity_filter(search_results)
        
        return diverse_results

    def _apply_diversity_filter(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity filtering to ensure variety in results."""
        if len(search_results) <= 10:
            return search_results  # No need to filter small result sets
        
        diverse_results = []
        location_count = {}
        type_count = {}
        price_ranges = {'low': 0, 'mid': 0, 'high': 0}
        
        for result in search_results:
            metadata = result['metadata']
            location = metadata.get('location', 'Unknown')
            prop_type = metadata.get('property_type', 'Unknown')
            price = metadata.get('price', 0)
            
            # Determine price range
            if price < 500000:
                price_range = 'low'
            elif price < 1500000:
                price_range = 'mid'
            else:
                price_range = 'high'
            
            # Check diversity constraints
            location_limit = 4  # Max 4 properties per location
            type_limit = 5      # Max 5 properties per type
            price_limit = 6     # Max 6 properties per price range
            
            if (location_count.get(location, 0) < location_limit and 
                type_count.get(prop_type, 0) < type_limit and 
                price_ranges.get(price_range, 0) < price_limit):
                
                diverse_results.append(result)
                location_count[location] = location_count.get(location, 0) + 1
                type_count[prop_type] = type_count.get(prop_type, 0) + 1
                price_ranges[price_range] = price_ranges.get(price_range, 0) + 1
            
            # Stop when we have enough diverse results
            if len(diverse_results) >= 15:
                break
        
        # If diversity filtering removed too many results, add back top-scored ones
        if len(diverse_results) < 8 and len(search_results) > len(diverse_results):
            remaining_results = [r for r in search_results if r not in diverse_results]
            diverse_results.extend(remaining_results[:8 - len(diverse_results)])
        
        return diverse_results

    def _is_property_related(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> bool:
        """Determine if query is property-related using simple classification."""
        try:
            # Prepare conversation context
            conversation_context = self._prepare_conversation_context(conversation_history)
            
            # Simplified classification prompt to avoid content filter
            classification_prompt = f"""Look at this conversation and current query. Is the user asking about real estate or properties?

{conversation_context}

Current Query: "{user_query}"

Answer only: YES or NO"""

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=5,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {str(e)}")
            # Fallback to simple keyword-based classification
            return self._is_property_query(user_query)

    def _generate_smart_response(self, user_query: str, search_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using LLM with conversation awareness and optimized context."""
        try:
            # Prepare optimized context from search results
            context = self._prepare_optimized_context(search_results, user_query)
            
            # Prepare conversation context
            conversation_context = self._prepare_conversation_context(conversation_history)
            
            # Enhanced system prompt with better instructions for smaller result sets
            system_prompt = """You are a helpful UAE real estate assistant. You help users find properties and answer questions about real estate in the UAE.

The properties provided have been pre-filtered and ranked for relevance. When responding:
- Analyze the provided properties carefully - they are already the most relevant matches
- For price range questions, examine all matching properties and provide accurate ranges
- For property type questions, focus on the relevant properties in the list
- Provide specific recommendations from the available properties
- Format prices with commas (e.g., 1,500,000 AED)
- Be conversational and helpful, highlighting the best matches
- If asked about specific features, focus on properties that have those features"""

            # Optimized user prompt with clearer instructions
            user_prompt = f"""User Query: "{user_query}"

{conversation_context}

Relevant Properties (pre-filtered and ranked by relevance):
{context}

Please provide a helpful response based on these properties. Focus on the most relevant matches and provide specific recommendations."""

            # Generate response using OpenAI with optimized parameters
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,  # Reduced from 800 since we have fewer, more relevant properties
                temperature=0.2  # Lower temperature for more focused responses
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating smart LLM response: {str(e)}")
            # Fallback to simple response
            return self._generate_fallback_response(user_query, search_results)

    def _prepare_optimized_context(self, search_results: List[Dict[str, Any]], user_query: str) -> str:
        """Prepare optimized context string from search results with relevance indicators."""
        if not search_results:
            return "No properties found in database."
        
        context_parts = []
        query_lower = user_query.lower()
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            relevance_score = result.get('final_relevance_score', result.get('similarity_score', 0))
            
            # Create enhanced property summary with relevance indicators
            property_info = f"Property {i}"
            
            # Add relevance indicator for top matches
            if relevance_score > 0.8:
                property_info += " ⭐ (Highly Relevant)"
            elif relevance_score > 0.6:
                property_info += " ✓ (Good Match)"
            
            property_info += f": {metadata.get('title', 'Untitled Property')}"
            property_info += f" | {metadata.get('property_type', 'Property')} in {metadata.get('location', 'Unknown')}"
            property_info += f" | {metadata.get('bedrooms', 0)} bed, {metadata.get('bathrooms', 0)} bath"
            property_info += f" | Price: {metadata.get('price', 0):,.0f} AED"
            
            if metadata.get('size', 0) > 0:
                property_info += f" | {metadata.get('size', 0):,.0f} sqft"
            
            # Add match reasons for highly relevant properties
            if relevance_score > 0.7:
                match_reasons = []
                if any(word in metadata.get('property_type', '').lower() for word in query_lower.split()):
                    match_reasons.append("type match")
                if any(word in metadata.get('location', '').lower() for word in query_lower.split()):
                    match_reasons.append("location match")
                if match_reasons:
                    property_info += f" | Matches: {', '.join(match_reasons)}"
            
            context_parts.append(property_info)
        
        # Add summary statistics for better LLM understanding
        if len(search_results) > 1:
            prices = [result['metadata'].get('price', 0) for result in search_results]
            min_price = min(prices)
            max_price = max(prices)
            context_parts.append(f"\nPrice Range Summary: {min_price:,.0f} - {max_price:,.0f} AED across {len(search_results)} properties")
        
        return "\n".join(context_parts)

    def _is_followup_query(self, user_query: str, conversation_history: List[Dict[str, str]]) -> bool:
        """Determine if this is a follow-up question referring to previous conversation."""
        if not conversation_history:
            return False
        
        query_lower = user_query.lower().strip()
        
        # Common follow-up indicators
        followup_patterns = [
            r'\b(these|those|them|it|this|that)\b',  # Reference words
            r'\b(show me more|tell me about|details about|what about)\b',  # Request for details
            r'\b(the first one|the second one|property \d+)\b',  # Specific property references
            r'\b(any others|more options|other properties)\b',  # Request for alternatives
            r'\b(yes|no|sure|okay|ok)\b',  # Confirmation/response words
            r'\b(which one|how much|where is)\b',  # Questions about previous results
        ]
        
        # Check if query contains reference patterns
        for pattern in followup_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # If query is very short and doesn't contain property keywords, likely a follow-up
        if len(query_lower.split()) <= 3 and not any(keyword in query_lower for keyword in ['find', 'search', 'show', 'villa', 'apartment']):
            return True
        
        return False

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract search filters from natural language query."""
        filters = {}
        query_lower = query.lower()
        
        # Extract bedrooms
        bedroom_patterns = [
            r'(\d+)[\s-]*bedroom',
            r'(\d+)[\s-]*bed(?:room)?s?',
            r'(\d+)[\s-]*br'
        ]
        for pattern in bedroom_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters['bedrooms'] = int(match.group(1))
                break
        
        # Extract bathrooms  
        bathroom_patterns = [
            r'(\d+)[\s-]*bathroom',
            r'(\d+)[\s-]*bath(?:room)?s?'
        ]
        for pattern in bathroom_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters['bathrooms'] = int(match.group(1))
                break
        
        # Extract price range
        price_patterns = [
            r'under\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m)?\s*aed',
            r'below\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m)?\s*aed',
            r'less\s+than\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m)?\s*aed',
            r'max\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m)?\s*aed',
            r'budget\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|m)?\s*aed'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                price_str = match.group(1).replace(',', '')
                price = float(price_str)
                
                # Check if it's in millions
                if 'million' in match.group(0) or 'm' in match.group(0):
                    price *= 1_000_000
                
                filters['max_price'] = price
                break
        
        # Extract property type
        property_types = ['villa', 'apartment', 'studio', 'penthouse', 'townhouse']
        for prop_type in property_types:
            if prop_type in query_lower:
                filters['property_type'] = prop_type.title()
                break
        
        # Extract location
        uae_locations = ['dubai', 'abu dhabi', 'sharjah', 'ajman', 'fujairah', 'ras al khaimah', 'umm al quwain']
        for location in uae_locations:
            if location in query_lower:
                filters['location'] = location.title()
                break
        
        return filters

    def _apply_filters(self, search_results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply extracted filters to search results."""
        filtered_results = []
        
        for result in search_results:
            metadata = result['metadata']
            matches = True
            
            # Filter by bedrooms
            if 'bedrooms' in filters:
                prop_bedrooms = metadata.get('bedrooms', 0)
                if prop_bedrooms != filters['bedrooms']:
                    matches = False
            
            # Filter by bathrooms
            if 'bathrooms' in filters:
                prop_bathrooms = metadata.get('bathrooms', 0) 
                if prop_bathrooms != filters['bathrooms']:
                    matches = False
            
            # Filter by max price
            if 'max_price' in filters:
                prop_price = metadata.get('price', 0)
                if prop_price > filters['max_price']:
                    matches = False
            
            # Filter by property type
            if 'property_type' in filters:
                prop_type = metadata.get('property_type', '').lower()
                filter_type = filters['property_type'].lower()
                if filter_type not in prop_type and prop_type not in filter_type:
                    matches = False
            
            # Filter by location
            if 'location' in filters:
                prop_location = metadata.get('location', '').lower()
                filter_location = filters['location'].lower()
                if filter_location not in prop_location:
                    matches = False
            
            if matches:
                filtered_results.append(result)
        
        return filtered_results

    def _format_criteria(self, filters: Dict[str, Any]) -> str:
        """Format criteria for user-friendly display."""
        criteria_parts = []
        
        if 'property_type' in filters:
            criteria_parts.append(f"{filters['property_type']}s")
        
        if 'bedrooms' in filters:
            criteria_parts.append(f"{filters['bedrooms']} bedroom{'s' if filters['bedrooms'] != 1 else ''}")
        
        if 'bathrooms' in filters:
            criteria_parts.append(f"{filters['bathrooms']} bathroom{'s' if filters['bathrooms'] != 1 else ''}")
        
        if 'location' in filters:
            criteria_parts.append(f"in {filters['location']}")
        
        if 'max_price' in filters:
            criteria_parts.append(f"under {filters['max_price']:,.0f} AED")
        
        if criteria_parts:
            return " ".join(criteria_parts)
        else:
            return "your search criteria"
    
    def _generate_response(self, user_query: str, search_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        """Legacy method - now just calls the smart response generator."""
        return self._generate_smart_response(user_query, search_results, conversation_history)

    def _prepare_conversation_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Prepare conversation context for the LLM."""
        if not conversation_history:
            return "Previous Conversation: None"
        
        # Get recent conversation (last 4 exchanges)
        recent_messages = conversation_history[-8:]
        
        context_parts = ["Previous Conversation:"]
        for message in recent_messages:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                # Truncate long assistant responses for context
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)

    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context string from search results."""
        if not search_results:
            return "No properties found in database."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            
            # Create clean property summary with title
            property_info = f"Property {i}: {metadata.get('title', 'Untitled Property')}"
            property_info += f" | {metadata.get('property_type', 'Property')} in {metadata.get('location', 'Unknown')}"
            property_info += f" | {metadata.get('bedrooms', 0)} bedrooms, {metadata.get('bathrooms', 0)} bathrooms"
            property_info += f" | Price: {metadata.get('price', 0):,.0f} AED"
            
            if metadata.get('size', 0) > 0:
                property_info += f" | Size: {metadata.get('size', 0):,.0f} sqft"
            
            context_parts.append(property_info)
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, user_query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate a simple fallback response when LLM fails."""
        if not search_results:
            return "I couldn't find any properties matching your criteria."
        
        total_properties = len(search_results)
        first_property = search_results[0]['metadata']
        
        response = f"I found {total_properties} propert{'y' if total_properties == 1 else 'ies'} matching your search. "
        
        if total_properties == 1:
            response += f"It's a {first_property.get('property_type', 'property').lower()} in {first_property.get('location', 'UAE')} "
            response += f"with {first_property.get('bedrooms', 0)} bedroom{'s' if first_property.get('bedrooms', 0) != 1 else ''} "
            response += f"and {first_property.get('bathrooms', 0)} bathroom{'s' if first_property.get('bathrooms', 0) != 1 else ''}, "
            response += f"priced at {first_property.get('price', 0):,.0f} AED."
        else:
            prices = [result['metadata'].get('price', 0) for result in search_results]
            min_price = min(prices)
            max_price = max(prices)
            response += f"Prices range from {min_price:,.0f} to {max_price:,.0f} AED. "
            response += "Would you like more details about any specific property?"
        
        return response
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            return {
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'vector_store': vector_stats,
                'model': self.model_name,
                'max_results': self.max_results,
                'similarity_threshold': self.similarity_threshold
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _is_property_query(self, user_query: str) -> bool:
        """Determine if the user query is asking about properties or just chatting."""
        query_lower = user_query.lower().strip()
        
        # Quick classification based on keywords and patterns
        property_keywords = [
            'property', 'properties', 'apartment', 'villa', 'house', 'home', 'real estate',
            'bedroom', 'bathroom', 'rent', 'buy', 'sale', 'price', 'aed', 'dirham',
            'dubai', 'abu dhabi', 'sharjah', 'ajman', 'location', 'area',
            'studio', 'penthouse', 'townhouse', 'sqft', 'square feet',
            'find', 'search', 'show', 'looking for', 'need', 'want'
        ]
        
        # Greetings and general conversation patterns
        general_keywords = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'thanks', 'thank you', 'bye', 'goodbye', 'help',
            'what can you do', 'who are you', 'what are you', 'how do you work'
        ]
        
        # If it's clearly a greeting or general question
        if any(keyword in query_lower for keyword in general_keywords):
            # Unless it also contains property keywords
            if not any(keyword in query_lower for keyword in property_keywords):
                return False
        
        # If it contains property-related keywords, it's likely a property query
        if any(keyword in query_lower for keyword in property_keywords):
            return True
        
        # For ambiguous cases, if it's a question, treat as property search
        if '?' in query_lower and len(query_lower.split()) > 3:
            return True
        
        # Short messages without property keywords are likely general conversation
        if len(query_lower.split()) <= 3:
            return False
        
        # Default to property search for longer, unclear queries
        return True
    
    def _handle_general_conversation(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Handle general conversation that's not about property search, with memory."""
        query_lower = user_query.lower().strip()
        
        # Greetings
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            return "Hello! 👋 I'm your UAE real estate assistant. I can help you find apartments, villas, and other properties across the UAE. What kind of property are you looking for today?"
        
        # Good morning/afternoon/evening
        if 'good morning' in query_lower:
            return "Good morning! ☀️ I'm here to help you find your perfect property in the UAE. Are you looking for a specific type of property or location?"
        
        if 'good afternoon' in query_lower:
            return "Good afternoon! 🌅 Ready to explore some amazing properties in the UAE? Tell me what you're looking for!"
        
        if 'good evening' in query_lower:
            return "Good evening! 🌆 I'd love to help you find your ideal property. What are your preferences?"
        
        # How are you
        if any(phrase in query_lower for phrase in ['how are you', 'how do you do']):
            return "I'm doing great, thank you for asking! 😊 I'm excited to help you discover some fantastic properties in the UAE. What can I help you find today?"
        
        # Thanks
        if any(word in query_lower for word in ['thank', 'thanks']):
            return "You're very welcome! 😊 I'm always happy to help with your property search. Is there anything else you'd like to know about UAE real estate?"
        
        # Goodbye
        if any(word in query_lower for word in ['bye', 'goodbye', 'see you']):
            return "Goodbye! 👋 It was great helping you with your property search. Feel free to come back anytime you need assistance finding properties in the UAE!"
        
        # What can you do / Help
        if any(phrase in query_lower for phrase in ['what can you do', 'help me', 'what do you do', 'how can you help']):
            return """I'm your UAE real estate assistant! 🏠 Here's what I can help you with:

• **Find Properties**: Search for apartments, villas, townhouses, penthouses, and studios
• **Filter by Preferences**: Bedrooms, bathrooms, price range, location
• **Location Search**: Properties in Dubai, Abu Dhabi, Sharjah, and other UAE cities  
• **Budget Planning**: Find properties within your price range
• **Property Details**: Get information about size, amenities, and features

Just tell me what you're looking for! For example:
- "Show me 2-bedroom apartments in Dubai"
- "Find villas under 2 million AED"
- "Properties in Dubai Marina with 3 bedrooms"

What would you like to search for?"""
        
        # Who are you / What are you
        if any(phrase in query_lower for phrase in ['who are you', 'what are you']):
            return "I'm an AI assistant specialized in UAE real estate! 🤖🏠 I have access to a database of properties across the UAE and can help you find exactly what you're looking for. Whether you want a cozy studio or a luxury villa, I'm here to help!"
        
        # Default friendly response for unclear general conversation
        return "I'm here to help you find amazing properties in the UAE! 🏠 Could you tell me more about what you're looking for? For example, are you interested in apartments, villas, or a specific location like Dubai or Abu Dhabi?"

    def set_performance_mode(self, mode: str = 'balanced'):
        """
        Set performance mode to optimize for speed vs accuracy.
        
        Args:
            mode: 'fast', 'balanced', or 'accurate'
        """
        if mode == 'fast':
            self.max_search_results = 10
            self.max_final_results = 8
            self.enable_diversity_filter = False
            self.enable_reranking = False
            logger.info("Set to FAST mode: Optimized for speed")
        elif mode == 'accurate':
            self.max_search_results = 30
            self.max_final_results = 20
            self.enable_diversity_filter = True
            self.enable_reranking = True
            logger.info("Set to ACCURATE mode: Optimized for accuracy")
        else:  # balanced
            self.max_search_results = 20
            self.max_final_results = 15
            self.enable_diversity_filter = True
            self.enable_reranking = True
            logger.info("Set to BALANCED mode: Balanced speed and accuracy")