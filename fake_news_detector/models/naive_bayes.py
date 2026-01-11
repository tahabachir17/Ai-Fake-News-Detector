
import logging
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
try:
    from fake_news_detector.models.evaluator import Evaluator # Assuming you might want to use a separate evaluator later, but for now strict implementation
    # Actually, let's keep it self-contained or use standard metrics
    pass
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NaiveBayesModel:
    """
    Naive Bayes Model with TF-IDF Vectorization for Fake News Detection.
    Features: TF-IDF with n-grams (1-3).
    """
    def __init__(self, ngram_range=(1, 3)):
        """
        Initialize the model pipeline.
        Args:
            ngram_range (tuple): The range of n-grams for TF-IDF.
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=50000)),
            ('clf', MultinomialNB())
        ])
        self.ngram_range = ngram_range

    def train(self, X_train, y_train):
        """
        Train the model.
        Args:
            X_train: Training text data.
            y_train: Training labels.
        """
        logging.info("Training Naive Bayes model...")
        self.model.fit(X_train, y_train)
        logging.info("Training complete.")

    def predict(self, X):
        """
        Predict labels for new data.
        Args:
            X: Input text data.
        Returns:
            Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        Args:
            X_test: Test text data.
            y_test: Test labels.
        Returns:
            dict: Dictionary containing accuracy and classification report.
        """
        logging.info("Evaluating model...")
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
        
        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        Args:
            filepath (str): Path to save the model.
        """
        logging.info(f"Saving model to {filepath}...")
        joblib.dump(self.model, filepath)
        logging.info("Model saved.")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        Args:
            filepath (str): Path to load the model from.
        """
        logging.info(f"Loading model from {filepath}...")
        self.model = joblib.load(filepath)
        logging.info("Model loaded.")
