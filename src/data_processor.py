import pandas as pd
import numpy as np
import os
import kaggle
from typing import List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data download, cleaning, and preparation for the RAG system."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_path = self.data_dir / "uae_real_estate_2024.csv"
        
    def download_dataset(self) -> bool:
        """Download the UAE Real Estate 2024 dataset from Kaggle."""
        try:
            # Set Kaggle API credentials from environment
            kaggle_username = os.getenv('KAGGLE_USERNAME')
            kaggle_key = os.getenv('KAGGLE_KEY')
            
            if not kaggle_username or not kaggle_key:
                logger.error("Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file")
                return False
            
            # Set environment variables for Kaggle API
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            
            # Set custom Kaggle base URL if provided
            kaggle_base_url = os.getenv('KAGGLE_BASE_URL')
            if kaggle_base_url:
                os.environ['KAGGLE_URL_BASE'] = kaggle_base_url
                logger.info(f"Using custom Kaggle base URL: {kaggle_base_url}")
                
            # Download dataset
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'mahmoudharidy/uae-real-estate-2024',
                path=str(self.data_dir),
                unzip=True
            )
            
            # Find the downloaded CSV file
            csv_files = list(self.data_dir.glob("*.csv"))
            if csv_files:
                # Rename to standard name
                csv_files[0].rename(self.dataset_path)
                logger.info(f"Dataset downloaded successfully to {self.dataset_path}")
                return True
            else:
                logger.error("No CSV file found after download")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the real estate dataset."""
        try:
            # Check if dataset exists, download if not
            if not self.dataset_path.exists():
                logger.info("Dataset not found. Attempting to download...")
                if not self.download_dataset():
                    # If download fails, create a sample dataset for testing
                    return self._create_sample_data()
            
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Clean and standardize the data
            df = self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            # Return sample data as fallback
            return self._create_sample_data()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataframe."""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize column names (common variations in real estate datasets)
        column_mapping = {
            'bedrooms': 'bedrooms',
            'bedroom': 'bedrooms',
            'bed': 'bedrooms',
            'bathrooms': 'bathrooms',
            'bathroom': 'bathrooms',
            'bath': 'bathrooms',
            'location': 'location',
            'city': 'location',
            'area': 'location',
            'price': 'price',
            'cost': 'price',
            'amount': 'price',
            'property_type': 'property_type',
            'type': 'property_type',
            'size': 'size',
            'area_sqft': 'size',
            'sqft': 'size'
        }
        
        # Apply column mapping (case insensitive)
        for old_col in df_clean.columns:
            for key, new_col in column_mapping.items():
                if key.lower() in old_col.lower():
                    df_clean = df_clean.rename(columns={old_col: new_col})
                    break
        
        # Ensure required columns exist
        required_columns = ['bedrooms', 'bathrooms', 'location', 'price', 'property_type']
        for col in required_columns:
            if col not in df_clean.columns:
                df_clean[col] = 'Not specified'
        
        # Clean numeric columns
        if 'bedrooms' in df_clean.columns:
            df_clean['bedrooms'] = pd.to_numeric(df_clean['bedrooms'], errors='coerce').fillna(0).astype(int)
        
        if 'bathrooms' in df_clean.columns:
            df_clean['bathrooms'] = pd.to_numeric(df_clean['bathrooms'], errors='coerce').fillna(0).astype(int)
        
        if 'price' in df_clean.columns:
            # Clean price column (remove currency symbols, convert to numeric)
            df_clean['price'] = df_clean['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce').fillna(0)
        
        # Clean text columns
        text_columns = ['location', 'property_type']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        
        # Remove rows with missing essential data
        df_clean = df_clean.dropna(subset=['location', 'price'])
        df_clean = df_clean[df_clean['price'] > 0]
        
        logger.info(f"Cleaned dataset: {len(df_clean)} rows remaining")
        return df_clean
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing when real dataset is not available."""
        logger.info("Creating sample dataset for testing...")
        
        locations = [
            'Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 
            'Fujairah', 'Umm Al Quwain', 'Dubai Marina', 'Downtown Dubai',
            'Business Bay', 'Jumeirah', 'Al Barsha', 'Deira', 'Bur Dubai'
        ]
        
        property_types = ['Apartment', 'Villa', 'Townhouse', 'Penthouse', 'Studio']
        
        # Property title templates for more realistic names
        title_templates = {
            'Apartment': ['Modern', 'Luxury', 'Spacious', 'Elegant', 'Contemporary', 'Premium', 'Stylish'],
            'Villa': ['Stunning', 'Magnificent', 'Exclusive', 'Private', 'Luxurious', 'Grand', 'Beautiful'],
            'Townhouse': ['Family', 'Modern', 'Spacious', 'Comfortable', 'Well-designed', 'Charming'],
            'Penthouse': ['Luxury', 'Sky', 'Premium', 'Exclusive', 'Panoramic', 'Elite', 'Prestigious'],
            'Studio': ['Cozy', 'Modern', 'Compact', 'Smart', 'Efficient', 'Urban', 'Contemporary']
        }
        
        sample_data = []
        for i in range(5000):  # Create 5000 sample properties
            property_type = np.random.choice(property_types)
            location = np.random.choice(locations)
            bedrooms = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.04, 0.01])
            
            # Generate a realistic property title
            title_prefix = np.random.choice(title_templates[property_type])
            if bedrooms == 0:
                bedroom_text = "Studio"
            else:
                bedroom_text = f"{bedrooms}-Bedroom"
            
            title = f"{title_prefix} {bedroom_text} {property_type} in {location}"
            
            sample_data.append({
                'property_id': f'PROP_{i+1:04d}',
                'title': title,
                'bedrooms': bedrooms,
                'bathrooms': np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.08, 0.02]),
                'location': location,
                'property_type': property_type,
                'price': np.random.randint(200000, 5000000),
                'size': np.random.randint(500, 5000),
                'description': f"Beautiful {property_type.lower()} in {location} with modern amenities."
            })
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample dataset with {len(df)} properties")
        return df
    
    def prepare_documents_for_indexing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare documents for vector indexing."""
        documents = []
        
        for _, row in df.iterrows():
            # Create a comprehensive text description for each property
            description = self._create_property_description(row)
            
            document = {
                'id': str(row.get('property_id', f"prop_{_}")),
                'text': description,
                'metadata': {
                    'title': str(row.get('title', f"Property {_+1}")),
                    'bedrooms': int(row.get('bedrooms', 0)),
                    'bathrooms': int(row.get('bathrooms', 0)),
                    'location': str(row.get('location', 'Unknown')),
                    'property_type': str(row.get('property_type', 'Unknown')),
                    'price': float(row.get('price', 0)),
                    'size': float(row.get('size', 0)) if pd.notna(row.get('size')) else 0,
                }
            }
            documents.append(document)
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents
    
    def _create_property_description(self, row: pd.Series) -> str:
        """Create a comprehensive text description for a property."""
        bedrooms = row.get('bedrooms', 0)
        bathrooms = row.get('bathrooms', 0)
        location = row.get('location', 'Unknown location')
        property_type = row.get('property_type', 'Property')
        price = row.get('price', 0)
        size = row.get('size', 0)
        
        # Create a rich, detailed description with keywords
        description_parts = []
        
        # Main property description
        bedroom_text = "studio" if bedrooms == 0 else f"{bedrooms} bedroom"
        if bedrooms > 1:
            bedroom_text += "s"
        
        bathroom_text = f"{bathrooms} bathroom"
        if bathrooms > 1:
            bathroom_text += "s"
        
        description_parts.append(f"This is a {property_type.lower()} property located in {location}")
        description_parts.append(f"The property features {bedroom_text} and {bathroom_text}")
        
        # Add price information
        if price > 0:
            if price < 500000:
                description_parts.append(f"Affordable property priced at {price:,.0f} AED")
            elif price < 1000000:
                description_parts.append(f"Moderately priced at {price:,.0f} AED")
            elif price < 2000000:
                description_parts.append(f"Premium property priced at {price:,.0f} AED")
            else:
                description_parts.append(f"Luxury property priced at {price:,.0f} AED")
        
        # Add size information
        if size > 0:
            if size < 1000:
                description_parts.append(f"Compact {size:,.0f} square feet")
            elif size < 2000:
                description_parts.append(f"Spacious {size:,.0f} square feet")
            else:
                description_parts.append(f"Large {size:,.0f} square feet")
        
        # Add location-specific keywords
        location_keywords = {
            'Dubai': 'modern city luxury metropolitan urban',
            'Abu Dhabi': 'capital city government district premium',
            'Sharjah': 'cultural emirate family friendly affordable',
            'Dubai Marina': 'waterfront marina luxury high-rise',
            'Downtown Dubai': 'central business district burj khalifa',
            'Business Bay': 'business district commercial towers',
            'Jumeirah': 'beachfront luxury residential exclusive'
        }
        
        if location in location_keywords:
            description_parts.append(f"Located in the {location_keywords[location]} area")
        
        # Add property type specific features
        type_features = {
            'Apartment': 'modern amenities building facilities community',
            'Villa': 'private garden parking independent house',
            'Townhouse': 'multi-level community shared amenities',
            'Penthouse': 'top floor panoramic views luxury exclusive',
            'Studio': 'compact efficient modern young professional'
        }
        
        if property_type in type_features:
            description_parts.append(f"Features include {type_features[property_type]}")
        
        # Add any additional description if available
        if 'description' in row and pd.notna(row['description']):
            description_parts.append(str(row['description']))
        
        # Add generic real estate keywords for better search
        description_parts.append("Real estate property for sale in UAE United Arab Emirates")
        
        return ". ".join(description_parts) 