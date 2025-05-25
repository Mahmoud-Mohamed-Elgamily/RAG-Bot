import chromadb
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector embeddings and similarity search using ChromaDB."""
    
    def __init__(self, persist_directory: str = "chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Collection name for properties
        self.collection_name = "uae_properties"
        
        try:
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "UAE Real Estate Properties"}
            )
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]], force_reindex: bool = False) -> bool:
        """Index documents in the vector store."""
        try:
            # Check if collection already has documents
            existing_count = self.collection.count()
            if existing_count > 0 and not force_reindex:
                logger.info(f"Collection already contains {existing_count} documents. Skipping indexing.")
                logger.info("Use force_reindex=True if you want to re-index the data.")
                return True
            elif existing_count > 0 and force_reindex:
                logger.info(f"Collection already contains {existing_count} documents. Force re-indexing...")
                
                # Delete the entire collection and recreate it
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info("Deleted existing collection")
                except Exception as delete_error:
                    logger.warning(f"Could not delete collection (may not exist): {delete_error}")
                
                # Recreate the collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "UAE Real Estate Properties"}
                )
                logger.info("Recreated collection for re-indexing")
            
            if not documents:
                logger.warning("No documents to index")
                return False
            
            # Prepare data for ChromaDB
            texts = [doc['text'] for doc in documents]
            ids = [doc['id'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Add documents to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                
                batch_texts = texts[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_embeddings = embeddings[i:batch_end].tolist()
                
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
                
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            total_count = self.collection.count()
            logger.info(f"Successfully indexed {total_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def search(self, query: str, n_results: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents based on query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build where clause for ChromaDB filtering."""
        conditions = []
        
        # Handle different filter types
        for key, value in filters.items():
            if key == 'min_price' and value is not None:
                conditions.append({'price': {'$gte': float(value)}})
            elif key == 'max_price' and value is not None:
                conditions.append({'price': {'$lte': float(value)}})
            elif key == 'bedrooms' and value is not None:
                conditions.append({'bedrooms': {'$eq': int(value)}})
            elif key == 'bathrooms' and value is not None:
                conditions.append({'bathrooms': {'$eq': int(value)}})
            elif key == 'location' and value:
                # Case-insensitive location search
                conditions.append({'location': {'$eq': str(value).title()}})
            elif key == 'property_type' and value:
                conditions.append({'property_type': {'$eq': str(value).title()}})
        
        # Handle price range (min and max)
        price_conditions = [c for c in conditions if 'price' in c]
        if len(price_conditions) > 1:
            # Combine min and max price into single condition
            price_condition = {'price': {}}
            for pc in price_conditions:
                price_condition['price'].update(pc['price'])
            # Remove individual price conditions and add combined one
            conditions = [c for c in conditions if 'price' not in c]
            conditions.append(price_condition)
        
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            # Use $and operator for multiple conditions
            return {'$and': conditions}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=min(100, count))
            
            stats = {
                'total_documents': count,
                'embedding_model': self.embedding_model_name,
                'collection_name': self.collection_name
            }
            
            if sample_results['metadatas']:
                # Analyze metadata to get insights
                locations = set()
                property_types = set()
                price_range = {'min': float('inf'), 'max': 0}
                bedroom_range = {'min': float('inf'), 'max': 0}
                
                for metadata in sample_results['metadatas']:
                    if 'location' in metadata:
                        locations.add(metadata['location'])
                    if 'property_type' in metadata:
                        property_types.add(metadata['property_type'])
                    if 'price' in metadata and metadata['price'] > 0:
                        price_range['min'] = min(price_range['min'], metadata['price'])
                        price_range['max'] = max(price_range['max'], metadata['price'])
                    if 'bedrooms' in metadata:
                        bedroom_range['min'] = min(bedroom_range['min'], metadata['bedrooms'])
                        bedroom_range['max'] = max(bedroom_range['max'], metadata['bedrooms'])
                
                stats.update({
                    'locations': sorted(list(locations)),
                    'property_types': sorted(list(property_types)),
                    'price_range': price_range if price_range['max'] > 0 else None,
                    'bedroom_range': bedroom_range if bedroom_range['max'] > 0 else None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'total_documents': 0, 'error': str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the entire collection and recreate it
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info("Deleted existing collection")
            except Exception as delete_error:
                logger.warning(f"Could not delete collection (may not exist): {delete_error}")
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "UAE Real Estate Properties"}
            )
            logger.info("Collection cleared and recreated successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False 