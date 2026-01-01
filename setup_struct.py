#!/usr/bin/env python3
"""
Fake News Detector Project Setup Script
Run: python setup_project.py
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define project structure
    structure = {
        "fake_news_detector": [
            # Core application
            "__init__.py",
            "main.py",
            
            # Configuration
            "config/__init__.py",
            "config/settings.py",
            "config/constants.py",
            
            # Data handling
            "data/__init__.py",
            "data/collector.py",
            "data/preprocessor.py",
            "data/loader.py",
            "data/raw/README.md",
            "data/processed/README.md",
            "data/external/README.md",
            
            # Feature engineering
            "features/__init__.py",
            "features/engineer.py",
            "features/extractors.py",
            
            # Models
            "models/__init__.py",
            "models/naive_bayes.py",
            "models/svm_model.py",
            "models/transformer.py",
            "models/ensemble.py",
            "models/trainer.py",
            "models/evaluator.py",
            "models/predictor.py",
            "models/saved_models/README.md",
            
            # API
            "api/__init__.py",
            "api/app.py",
            "api/routes/__init__.py",
            "api/routes/predict.py",
            "api/routes/health.py",
            "api/routes/docs.py",
            "api/middleware/__init__.py",
            "api/middleware/auth.py",
            "api/middleware/rate_limit.py",
            
            # Utils
            "utils/__init__.py",
            "utils/logger.py",
            "utils/helpers.py",
            "utils/validators.py",
            
            # Notebooks for exploration
            "notebooks/01_data_exploration.ipynb",
            "notebooks/02_baseline_models.ipynb",
            "notebooks/03_advanced_models.ipynb",
            "notebooks/04_model_ensembling.ipynb",
            
            # Tests
            "tests/__init__.py",
            "tests/test_data.py",
            "tests/test_models.py",
            "tests/test_api.py",
            "tests/test_utils.py",
            
            # Web interface (optional)
            "web/__init__.py",
            "web/templates/index.html",
            "web/templates/results.html",
            "web/static/css/style.css",
            "web/static/js/main.js",
            
            # Monitoring
            "monitoring/__init__.py",
            "monitoring/dashboard.py",
            "monitoring/metrics.py",
            
            # Documentation
            "docs/README.md",
            "docs/api.md",
            "docs/deployment.md",
            
            # Configuration files
            ".env.example",
            ".gitignore",
            "requirements.txt",
            "requirements-dev.txt",
            "setup.py",
            "pyproject.toml",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
            "README.md",
            "LICENSE",
            
            # CI/CD
            ".github/workflows/ci.yml",
            ".github/workflows/cd.yml",
            
            # ML Experiment tracking
            "mlruns/README.md",
            
            # Scripts
            "scripts/setup.sh",
            "scripts/train_model.sh",
            "scripts/run_api.sh",
            "scripts/deploy.sh",
        ]
    }
    
    # Create directories and files
    for base_dir, items in structure.items():
        for item in items:
            path = Path(base_dir) / item
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist
            if not path.exists():
                if path.suffix:  # It's a file
                    with open(path, 'w', encoding='utf-8') as f:
                        # Add basic content to certain files
                        if path.name == '__init__.py':
                            f.write('"""Module initialization."""\n')
                        elif path.name == 'README.md':
                            f.write(f'# {path.parent.name}\n\nDirectory for {path.parent.name}.\n')
                        elif path.name == 'requirements.txt':
                            f.write('''# Core dependencies
scikit-learn>=1.0.0
transformers>=4.25.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
spacy>=3.7.0
nltk>=3.8.0
python-dotenv>=1.0.0
joblib>=1.3.0
''')
                        elif path.name == 'main.py':
                            f.write('''#!/usr/bin/env python3
"""
Fake News Detector - Main Entry Point
"""

import argparse
from fake_news_detector.models.predictor import FakeNewsPredictor

def main():
    parser = argparse.ArgumentParser(description='Fake News Detector CLI')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--model', type=str, default='ensemble', 
                       choices=['naive_bayes', 'svm', 'ensemble'], 
                       help='Model to use for prediction')
    
    args = parser.parse_args()
    
    predictor = FakeNewsPredictor()
    
    if args.text:
        result = predictor.predict(args.text, model_type=args.model)
        print(f"Result: {result}")
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        result = predictor.predict(text, model_type=args.model)
        print(f"Result: {result}")
    else:
        print("Please provide either --text or --file argument")

if __name__ == '__main__':
    main()
''')
                        elif path.name == 'config/settings.py':
                            f.write('''"""
Configuration settings for the Fake News Detector
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
LOG_DIR = BASE_DIR / 'logs'

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_CONFIG = {
    'naive_bayes': {
        'save_path': MODELS_DIR / 'naive_bayes.joblib',
        'features': ['tfidf', 'ngrams'],
    },
    'svm': {
        'save_path': MODELS_DIR / 'svm.joblib',
        'features': ['tfidf', 'sentiment', 'readability'],
    },
    'transformer': {
        'model_name': 'distilbert-base-uncased',
        'save_path': MODELS_DIR / 'transformer',
    }
}

# API settings
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', '8000')),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true',
    'workers': int(os.getenv('API_WORKERS', '1')),
}

# Data settings
DATA_CONFIG = {
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
}

# Logging
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}
''')
                print(f"Created: {path}")
            else:
                print(f"Exists: {path}")

if __name__ == '__main__':
    print("Setting up Fake News Detector project structure...")
    create_project_structure()
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. cd fake_news_detector")
    print("2. python -m venv venv")
    print("3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("4. pip install -r requirements.txt")
    print("5. python main.py --text \"Your news text here\"")