import os
import joblib
import logging
import pandas as pd
import numpy as np

from fake_news_detector.data.preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FakeNewsPredictor:
    """
    Load trained models and make predictions on new data.
    
    Supports:
        - Raw text input
        - URL input (auto-detected, article text scraped)
    """

    def __init__(self, model_name='naive_bayes'):
        self.model_name = model_name
        self.model = None
        self.models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        self._preprocessor = TextPreprocessor()
        self.load_model()

    def load_model(self):
        """Loads the specified model from disk."""
        try:
            if self.model_name == 'naive_bayes':
                model_path = os.path.join(self.models_dir, 'best_naive_bayes.pkl')
                if os.path.exists(model_path):
                    logging.info(f"Loading Naive Bayes model from {model_path}...")
                    self.model = joblib.load(model_path)
                    logging.info("Model loaded successfully.")
                else:
                    logging.warning(
                        f"Model file not found at {model_path}. "
                        "Please train the model first."
                    )
            elif self.model_name in ['svm', 'transformer']:
                logging.warning(
                    f"Model '{self.model_name}' is not yet implemented. "
                    "Please select 'naive_bayes'."
                )
            else:
                logging.error(f"Unknown model name: {self.model_name}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def predict(self, text):
        """
        Predict whether the input text is REAL or FAKE.
        
        This method accepts raw (uncleaned) text or a URL. It handles 
        preprocessing internally.

        Args:
            text (str): Raw article text or a URL.

        Returns:
            dict: {
                'label': str,
                'score': float,
                'probabilities': dict,
                'input_type': str       # 'url' or 'text'
            }
        """
        if self.model is None:
            return {
                'label': 'ERROR',
                'score': 0.0,
                'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                'input_type': 'unknown',
                'message': "Model not loaded. Please train the model or check the path."
            }

        try:
            # Detect input type and preprocess
            input_type = 'url' if TextPreprocessor.is_url(text) else 'text'
            cleaned_text = self._preprocessor.process_input(text)

            # Predict
            prediction = self.model.predict([cleaned_text])[0]
            probabilities = self.model.predict_proba([cleaned_text])[0]

            # Map classes to labels
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
