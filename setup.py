#!/usr/bin/env python3
"""
Setup script for UAE Real Estate RAG Chatbot
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    """Setup environment file."""
    env_file = Path(".env")
    example_file = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if example_file.exists():
        print("\n📝 Creating .env file from template...")
        import shutil
        shutil.copy(example_file, env_file)
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your API keys before running the application")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        "streamlit",
        "openai", 
        "pandas",
        "numpy",
        "chromadb",
        "sentence_transformers"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["data", "chroma_db", "src"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n🔧 Checking environment variables...")
    
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    required_vars = ["AZURE_OPENAI_API_KEY"]
    optional_vars = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (optional)")
            missing_optional.append(var)
    
    if missing_required:
        print(f"\n❌ Missing required variables: {', '.join(missing_required)}")
        print("Please edit your .env file with the required API keys")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional variables: {', '.join(missing_optional)}")
        print("The system will use sample data if Kaggle credentials are not provided")
    
    return True

def main():
    """Main setup function."""
    print("🏠 UAE Real Estate RAG Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Check environment variables
    env_ok = check_environment_variables()
    
    print("\n" + "=" * 50)
    if env_ok:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Ensure your .env file has the correct API keys")
        print("2. Run: streamlit run streamlit_app.py")
        print("3. Click 'Initialize System' in the sidebar")
        print("4. Start chatting about UAE real estate!")
    else:
        print("⚠️  Setup completed with warnings")
        print("\nNext steps:")
        print("1. Edit your .env file with the required API keys")
        print("2. Run this setup script again to verify")
        print("3. Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 