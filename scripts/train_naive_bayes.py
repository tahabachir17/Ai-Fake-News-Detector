
import sys
import os
import logging
from sklearn.model_selection import train_test_split

# Add the project root to the python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fake_news_detector.data.loader import DataLoader
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.models.naive_bayes import NaiveBayesModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Load Data
    loader = DataLoader()  # Defaults to fake_news_detector/data/raw
    df = loader.load_data()
    
    if df is None or df.empty:
        logging.error("Failed to load data. Exiting.")
        return

    # 2. Preprocess Data
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)

    # 3. Split Data
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Test set size: {len(X_test)}")

    # 4. Train Model
    model = NaiveBayesModel(ngram_range=(1, 3))
    model.train(X_train, y_train)

    # 5. Evaluate Model
    results = model.evaluate(X_test, y_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    logging.info(f"Confusion Matrix:\n{cm}")

    # 6. Save Model
    models_dir = os.path.join(
        os.path.dirname(__file__), '..', 
        'fake_news_detector', 'models', 'saved_models'
    )
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_naive_bayes.pkl')
    
    model.save_model(model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
