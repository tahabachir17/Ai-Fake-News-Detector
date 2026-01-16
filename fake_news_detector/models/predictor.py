import os
import joblib
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FakeNewsPredictor:
    """
    Class to load trained models and make predictions on new data.
    """
    def __init__(self, model_name='naive_bayes'):
        self.model_name = model_name
        self.model = None
        self.models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        self.load_model()

    def load_model(self):
        """
        Loads the specified model from disk.
        """
        try:
            if self.model_name == 'naive_bayes':
                model_path = os.path.join(self.models_dir, 'best_naive_bayes.pkl')
                if os.path.exists(model_path):
                    logging.info(f"Loading Naive Bayes model from {model_path}...")
                    self.model = joblib.load(model_path)
                    logging.info("Model loaded successfully.")
                else:
                    logging.warning(f"Model file not found at {model_path}. Please train the model first.")
            elif self.model_name in ['svm', 'transformer']:
                 logging.warning(f"Model '{self.model_name}' is not yet implemented. Please select 'naive_bayes'.")
            else:
                logging.error(f"Unknown model name: {self.model_name}")

        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def predict(self, text):
        """
        Predicts whether the input text is REAL or FAKE.

        Args:
            text (str or pd.Series): Input text.

        Returns:
            dict: Dictionary containing prediction label, score, and probabilities.
        """
        if self.model is None:
             return {
                'label': 'ERROR',
                'score': 0.0,
                'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                'message': "Model not loaded. Please train the model or check the path."
            }

        try:
            # Ensure text is in the right format (list or Series) 
            # The model pipeline (tfidf) typically expects an iterable of strings
            if isinstance(text, str):
                text_input = [text]
            else:
                text_input = text
            
            # Predict
            prediction = self.model.predict(text_input)[0]
            probabilities = self.model.predict_proba(text_input)[0]
            
            # Map classes to labels (assuming 0=FAKE, 1=REAL or similar, need to verify from training)
            # Standard convention in this dataset often varies, but usually:
            # Let's assume the model classes are ['FAKE', 'REAL'] or [0, 1].
            # If classes_ is present, we can map correctly.
            
            classes = self.model.classes_
            prob_dict = {str(cls): prob for cls, prob in zip(classes, probabilities)}
            
            # meaningful label
            label = str(prediction)
            
            # Confidence score (max probability)
            score = max(probabilities)
            
            return {
                'label': label,
                'score': score,
                'probabilities': prob_dict
            }

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {
                'label': 'ERROR',
                'score': 0.0,
                'probabilities': {},
                'message': str(e)
            }
