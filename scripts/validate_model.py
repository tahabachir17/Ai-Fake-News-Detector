import sys
import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.loader import DataLoader
from fake_news_detector.data.preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ACCURACY_THRESHOLD = 0.92
# F1_THRESHOLD could be dynamic, but for now we'll log it.
# In a real scenario, we'd fetch the previous production metrics from a store.

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def validate_model():
    logging.info("Starting Model Validation...")

    # 1. Load Data
    data_loader = DataLoader() # Assuming DataLoader can fetch distinct test set or we split again
    # In a real CI env, we might have a specific fixed 'golden' dataset file.
    # For now, we'll re-use the loader's fetching mechanism and split.
    # Ideally: raw_data = pd.read_csv('tests/data/golden_dataset.csv')
    raw_data = data_loader.fetch_raw_data()
    
    if raw_data is None or raw_data.empty:
        logging.error("No data found for validation!")
        sys.exit(1)

    # 2. Preprocess
    logging.info("Preprocessing data...")
    preprocessor = TextPreprocessor()
    preprocessor.fit(raw_data) # Note: In prod validity, we should load fitted preprocessor too!
    cleaned_series = preprocessor.transform(raw_data)
    
    processed_df = pd.DataFrame({
        'text': cleaned_series,
        'label': raw_data['label']
    })

    # 3. Split (Using a fixed seed for reproducibility in validation if possible)
    # Using the same split as training might be cheating if we don't have separate holdout.
    # We will assume the model was trained on a different execution or we are validating on the test split here.
    _, X_test, _, y_test = data_loader.get_train_test_split(
        processed_df, 
        test_size=0.2, 
        target_column='label'
    )

    # 4. Load Model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'fake_news_detector', 'models', 'saved_models', 'best_model.pkl')
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 5. Evaluate
    predictions = model.predict(X_test['text'])
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    logging.info(f"Validation Metric - Accuracy: {accuracy:.4f}")
    logging.info(f"Validation Metric - F1-Score: {f1:.4f}")

    if accuracy < ACCURACY_THRESHOLD:
        logging.error(f"Model Validation FAILED: Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}")
        sys.exit(1)
    
    # Placeholder for F1 comparison (requires previous state)
    # previous_f1 = get_previous_f1() 
    # if f1 < previous_f1:
    #     logging.error("Model Validation FAILED: F1-score dropped.")
    #     sys.exit(1)

    logging.info("Model Validation PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    validate_model()
