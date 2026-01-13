import re
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords as nltk_stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Text preprocessing transformer compatible with Scikit-Learn pipelines.
    
    Arguments:
        stopwords (list, str or None): List of stopwords to remove, or 'english' to use NLTK's english stopwords.
                                     If None, no stopwords are removed.
    """
    def __init__(self, stopwords='english'):
        self.stopwords = stopwords
        self._final_stopwords = set()

    def fit(self, X, y=None):
        """
        Fit method to prepare stopwords.
        """
        if self.stopwords == 'english':
            try:
                self._final_stopwords = set(nltk_stopwords.words('english'))
            except LookupError:
                logging.info("Downloading NLTK stopwords...")
                nltk.download('stopwords')
                self._final_stopwords = set(nltk_stopwords.words('english'))
        elif isinstance(self.stopwords, list):
            self._final_stopwords = set(self.stopwords)
        else:
            self._final_stopwords = set()
            
        return self

    def transform(self, X):
        """
        Transforms the input data by applying text cleaning operations.
        
        Args:
            X (pd.DataFrame, pd.Series, or list): Input text data.
            
        Returns:
            pd.Series: Series containing cleaned text.
        """
        # Input handling and conversion to Series
        if isinstance(X, list):
            X = pd.Series(X)
        elif isinstance(X, pd.DataFrame):
            # If explicit 'text' column exists, use it
            if 'text' in X.columns:
                # If 'title' also exists, user might want to combine them, but for a standard Transformer
                # we usually expect the input to be ready or handle specific logic.
                # Given previous code used title+text, let's support that if both exist.
                if 'title' in X.columns:
                    logging.info("Combining title and text columns...")
                    X = X['title'].astype(str) + " " + X['text'].astype(str)
                else:
                    X = X['text']
            elif X.shape[1] == 1:
                X = X.iloc[:, 0]
            else:
                # Fallback: assume all columns are text and concat? or just fail.
                # Let's take the first column to be safe if 'text' isn't there, 
                # but valid ML pipelines usually select columns before passing here.
                logging.warning("No 'text' column found. Using the first column as input.")
                X = X.iloc[:, 0]
        
        # Ensure X is a Series
        if not isinstance(X, pd.Series):
             X = pd.Series(X)

        # Make a copy and ensure string type
        X_clean = X.copy().astype(str)
        
        logging.info("Starting text cleaning...")

        # 1. Lowercasing (Vectorized)
        X_clean = X_clean.str.lower()
        
        # 2. Removing URLs and HTML tags (Vectorized)
        # URL regex
        X_clean = X_clean.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)
        # HTML tag regex
        X_clean = X_clean.str.replace(r'<.*?>', ' ', regex=True)
        
        # 3. Removing special characters/punctuation (Vectorized)
        # Keep word characters and whitespace
        X_clean = X_clean.str.replace(r'[^\w\s]', '', regex=True)
        
        # 4. Removing stopwords
        if self._final_stopwords:
            # Note: .apply with a python function is not strictly "vectorized" in the C sense,
            # but it is the standard Pandas approach for token-level operations like stopword removal.
            # Strictly vectorized string replacement for all stopwords would require a massive regex,
            # which can be slower and hit recursion limits.
            logging.info("Removing stopwords...")
            stop_words = self._final_stopwords
            X_clean = X_clean.apply(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))
            
        # 5. Remove extra whitespace (Vectorized)
        X_clean = X_clean.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        logging.info("Text cleaning complete.")
        return X_clean
