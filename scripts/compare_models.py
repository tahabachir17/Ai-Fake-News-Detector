
import sys
import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.loader import DataLoader
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.models.naive_bayes import NaiveBayesModel
from fake_news_detector.models.passive_aggressive import PassiveAggressiveModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a model and returns metrics.
    """
    predictions = model.predict(X_test)
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, predictions, average='weighted', zero_division=0)
    }
    return metrics

def main():
    logging.info("Starting Model Comparison...")

    # 1. Load Data
    data_loader = DataLoader()
    raw_data = data_loader.fetch_raw_data()
    
    if raw_data is None or raw_data.empty:
        logging.error("No data found!")
        return

    # 2. Preprocess
    logging.info("Preprocessing data...")
    preprocessor = TextPreprocessor()
    preprocessor.fit(raw_data)
    cleaned_series = preprocessor.transform(raw_data)
    
    processed_df = pd.DataFrame({
        'text': cleaned_series,
        'label': raw_data['label']
    })

    # 3. Split Data
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        processed_df, 
        test_size=0.2, 
        target_column='label'
    )

    # 4. Train and Evaluate Models
    results = []
    
    # Naive Bayes
    logging.info("Training Naive Bayes...")
    nb_model = NaiveBayesModel()
    nb_model.train(X_train['text'], y_train)
    nb_metrics = evaluate_model(nb_model, X_test['text'], y_test, "Naive Bayes")
    results.append(nb_metrics)
    
    # Passive Aggressive
    logging.info("Training Passive Aggressive Classifier...")
    pa_model = PassiveAggressiveModel()
    pa_model.train(X_train['text'], y_train)
    pa_metrics = evaluate_model(pa_model, X_test['text'], y_test, "Passive Aggressive")
    results.append(pa_metrics)
    
    # 5. Compare Results
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))
    
    # 6. Save Best Model
    best_model_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    logging.info(f"Best Model: {best_model_name}")
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'fake_news_detector', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    if best_model_name == "Naive Bayes":
        best_model = nb_model
    else:
        best_model = pa_model
        
    save_path = os.path.join(models_dir, 'best_model.pkl')
    best_model.save_model(save_path)
    logging.info(f"Saved best model to {save_path}")

if __name__ == "__main__":
    main()
