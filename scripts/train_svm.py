import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.loader import DataLoader
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.models.svm_model import SVMModel
from fake_news_detector.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting SVM Training Pipeline...")

    # 1. Load data
    data_loader = DataLoader()
    raw_data = data_loader.fetch_raw_data()

    if raw_data is None or raw_data.empty:
        logging.error("No data found! Please check the 'fake_news_detector/data/raw' directory.")
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

    # 3. Train/test split
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        processed_df,
        test_size=0.2,
        target_column='label'
    )

    # 4. Initialize and optimize SVM model
    logging.info("Initializing and optimizing SVM Model...")
    svm_model = SVMModel()

    # Hyperparameter grid
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__min_df': [1, 2],
        'clf__estimator__alpha': [1e-4, 1e-3, 1e-2]
    }

    grid_search = GridSearchCV(
        svm_model.model,
        param_grid,
        cv=3,
        n_jobs=1,       # Avoid pickling/memory issues on Windows
        verbose=1,
        scoring='f1_weighted'
    )

    logging.info("Running GridSearchCV...")
    grid_search.fit(X_train['text'], y_train)

    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    # Update model with best estimator
    svm_model.model = grid_search.best_estimator_

    # 5. Evaluate Model
    logging.info("Evaluating best SVM model on test set...")
    predictions = svm_model.predict(X_test['text'])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, predictions)
    print("\nSVM Model Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Visualizations and reports
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'fake_news_detector', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    cm_path = os.path.join(models_dir, 'svm_confusion_matrix.png')
    evaluator.plot_confusion_matrix(y_test, predictions, save_path=cm_path)

    report_path = os.path.join(models_dir, 'svm_classification_report.txt')
    evaluator.save_classification_report(y_test, predictions, save_path=report_path)

    # 6. Save best model
    model_path = os.path.join(models_dir, 'best_svm.pkl')
    svm_model.save_model(model_path)

    logging.info("SVM Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
