import logging
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from fake_news_detector.data.preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SVMModel:
    """
    Linear SVM Model with TF-IDF Vectorization for Fake News Detection.

    Uses SGDClassifier(loss='hinge') for efficient linear SVM training,
    wrapped in CalibratedClassifierCV for probability estimates.

    Features:
        - TF-IDF with configurable n-grams
        - Integrated TextPreprocessor for end-to-end inference
        - Supports both raw text and URL inputs
        - Probability calibration via Platt scaling
    """

    def __init__(self, ngram_range=(1, 2), max_features=50000, max_iter=1000):
        """
        Initialize the SVM model pipeline.

        Args:
            ngram_range (tuple): The range of n-grams for TF-IDF.
            max_features (int): Maximum number of TF-IDF features.
            max_iter (int): Maximum iterations for SGD convergence.
        """
        base_svm = SGDClassifier(
            loss='hinge',
            max_iter=max_iter,
            tol=1e-3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words='english',
                max_features=max_features,
                sublinear_tf=True   # log(1 + tf) — improves SVM performance on text
            )),
            ('clf', CalibratedClassifierCV(base_svm, cv=3))
        ])
        self.ngram_range = ngram_range
        self._preprocessor = TextPreprocessor()

    def train(self, X_train, y_train):
        """
        Train the model on preprocessed text data.

        Args:
            X_train: Training text data (iterable of strings).
            y_train: Training labels.
        """
        logging.info("Training SVM model...")
        self.model.fit(X_train, y_train)
        logging.info("SVM training complete.")

    def predict(self, X):
        """
        Predict labels for preprocessed text data.

        Args:
            X: Input text data (iterable of strings).

        Returns:
            numpy.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Return class probability estimates for preprocessed text data.

        Probabilities are calibrated via CalibratedClassifierCV (Platt scaling).

        Args:
            X: Input text data (iterable of strings).

        Returns:
            numpy.ndarray: Probability estimates (shape: n_samples x n_classes).
        """
        return self.model.predict_proba(X)

    def predict_from_input(self, user_input: str) -> dict:
        """
        End-to-end prediction from raw text or a URL.

        Handles preprocessing internally:
            - If input is a URL → scrapes article text first
            - Applies text cleaning
            - Runs TF-IDF + SVM prediction

        Args:
            user_input (str): Raw article text or a URL.

        Returns:
            dict: {
                'label': str,           # Predicted class label
                'score': float,         # Confidence (max probability)
                'probabilities': dict,  # {class: probability, ...}
                'input_type': str       # 'url' or 'text'
            }
        """
        try:
            input_type = 'url' if self._preprocessor.is_url(user_input) else 'text'
            cleaned_text = self._preprocessor.process_input(user_input)

            prediction = self.model.predict([cleaned_text])[0]
            probabilities = self.model.predict_proba([cleaned_text])[0]

            classes = self.model.classes_
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

            return {
                'label': str(prediction),
                'score': float(max(probabilities)),
                'probabilities': prob_dict,
                'input_type': input_type
            }

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {
                'label': 'ERROR',
                'score': 0.0,
                'probabilities': {},
                'input_type': 'unknown',
                'message': str(e)
            }

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on preprocessed test data.

        Args:
            X_test: Test text data (iterable of strings).
            y_test: Test labels.

        Returns:
            dict: Dictionary containing accuracy and classification report.
        """
        logging.info("Evaluating SVM model...")
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        logging.info(f"SVM Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")

        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def save_model(self, filepath: str):
        """
        Save the trained model pipeline to disk.

        Args:
            filepath (str): Destination path.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        logging.info(f"Saving SVM model to {filepath}...")
        joblib.dump(self.model, filepath)
        logging.info("SVM model saved.")

    def load_model(self, filepath: str):
        """
        Load a trained model pipeline from disk.

        Args:
            filepath (str): Path to the saved model.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        logging.info(f"Loading SVM model from {filepath}...")
        self.model = joblib.load(filepath)
        logging.info("SVM model loaded.")
