import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.loader import DataLoader
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.models.naive_bayes import NaiveBayesModel
from fake_news_detector.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Fake News Detector Training Pipeline...")

    # 1. Initialize DataCollector (via DataLoader) and fetch data
    data_loader = DataLoader()
    raw_data = data_loader.fetch_raw_data()
    
    if raw_data is None or raw_data.empty:
        logging.error("No data found! Please check the 'fake_news_detector/data/raw' directory.")
        return

    # 2. Text Preprocessing
    logging.info("Preprocessing data...")
    preprocessor = TextPreprocessor()
    # Fit preprocessor (loads stopwords)
    preprocessor.fit(raw_data)
    
    # We need to clean the text before splitting if we want consistent cleaning.
    # However, to use the Transformer properly in a pipeline, we usually put it IN the pipeline.
    # But here, the requirement says "Use TextPreprocessor to clean the data" *then* split.
    # Also TextPreprocessor returns a Series, so we need to put it back in the DF or careful with alignment.
    
    # Let's clean and replace the 'text' column for simplicity in this script.
    # Note: transforming the whole dataset before split is okay for stateless cleaning (regex),
    # but strictly speaking should be done after split for things like vocabulary (which is in TF-IDF, not here).
    
    # We will use the 'text' and 'title' combination logic inside TextPreprocessor.transform
    # But TextPreprocessor.transform returns a Series.
    # Let's assume we want to use the combined/cleaned text as our 'text' feature.
    
    cleaned_series = preprocessor.transform(raw_data)
    # create a new dataframe for splitting to keep y aligned
    processed_df = pd.DataFrame({
        'text': cleaned_series,
        'label': raw_data['label']
    })
    
    # 3. Data Splitting
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        processed_df, 
        test_size=0.2, 
        target_column='label'
    )
    
    # 4. Initialize and Optimize Naive Bayes Model
    logging.info("Initializing and optimizing Naive Bayes Model...")
    nb_model = NaiveBayesModel()
    
    # Hyperparameter Grid
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)], # Removed (1, 3) to save memory
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__min_df': [1, 2],
        'clf__alpha': [0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(
        nb_model.model, 
        param_grid, 
        cv=3, 
        n_jobs=1, # Set to 1 to avoid PicklingError/MemoryError on Windows with large data
        verbose=1,
        scoring='f1_weighted' # optimize for balanced performance
    )
    
    logging.info("Running GridSearchCV...")
    # TfidfVectorizer expects an iterable of strings, not a DataFrame. 
    # X_train is a DataFrame, so we need to select the 'text' column.
    grid_search.fit(X_train['text'], y_train)
    
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    
    # Update model with best estimator
    nb_model.model = grid_search.best_estimator_
    
    # 5. Evaluate Model
    logging.info("Evaluating best model on test set...")
    predictions = nb_model.predict(X_test['text'])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, predictions)
    print("\nModel Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Visualizations and Reports
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'fake_news_detector', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(y_test, predictions, save_path=cm_path)
    
    report_path = os.path.join(models_dir, 'classification_report.txt')
    evaluator.save_classification_report(y_test, predictions, save_path=report_path)
    
    # 6. Save Best Model
    model_path = os.path.join(models_dir, 'best_naive_bayes.pkl')
    nb_model.save_model(model_path)
    
    logging.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
