import re
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Text preprocessing transformer compatible with Scikit-Learn pipelines.
    
    Handles:
        - Cleaning raw training DataFrames (preprocess_dataframe)
        - Cleaning individual text or URL input (process_input)
        - Sklearn fit/transform interface for pipeline integration
    
    Arguments:
        stopwords (list, str or None): List of stopwords to remove, or 'english' 
                                       to use NLTK's english stopwords.
                                       If None, no stopwords are removed.
    """
    def __init__(self, stopwords='english'):
        self.stopwords = stopwords
        self._final_stopwords = set()

    def fit(self, X=None, y=None):
        """
        Fit method to prepare stopwords.
        """
        if self.stopwords == 'english':
            try:
                self._final_stopwords = set(nltk_stopwords.words('english'))
            except LookupError:
                logging.info("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
                self._final_stopwords = set(nltk_stopwords.words('english'))
        elif isinstance(self.stopwords, list):
            self._final_stopwords = set(self.stopwords)
        else:
            self._final_stopwords = set()
            
        return self

    # ------------------------------------------------------------------ #
    #  Core text cleaning (vectorized on pd.Series)                       #
    # ------------------------------------------------------------------ #

    def _clean_series(self, series: pd.Series) -> pd.Series:
        """
        Applies all text cleaning steps to a pandas Series.
        
        Steps:
            1. Lowercase
            2. Remove URLs and HTML tags
            3. Remove special characters / punctuation
            4. Remove stopwords
            5. Collapse extra whitespace
        """
        s = series.copy().astype(str)

        # 1. Lowercasing
        s = s.str.lower()

        # 2. Remove URLs and HTML tags
        s = s.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)
        s = s.str.replace(r'<.*?>', ' ', regex=True)

        # 3. Remove special characters / punctuation (keep word chars + spaces)
        s = s.str.replace(r'[^\w\s]', '', regex=True)
        # Remove digits as well (optional but common for news classification)
        s = s.str.replace(r'\d+', '', regex=True)

        # 4. Remove stopwords
        if self._final_stopwords:
            logging.info("Removing stopwords...")
            stop_words = self._final_stopwords
            s = s.apply(
                lambda text: ' '.join(
                    w for w in text.split() if w not in stop_words
                )
            )

        # 5. Collapse extra whitespace
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

        return s

    # ------------------------------------------------------------------ #
    #  Sklearn Transformer interface (fit / transform)                    #
    # ------------------------------------------------------------------ #

    def transform(self, X):
        """
        Transforms the input data by applying text cleaning operations.
        
        Args:
            X (pd.DataFrame, pd.Series, list, or str): Input text data.
            
        Returns:
            pd.Series: Series containing cleaned text.
        """
        # Ensure stopwords are ready (allows transform without explicit fit)
        if not self._final_stopwords and self.stopwords:
            self.fit()

        # --- Convert input to Series ---
        if isinstance(X, str):
            X = pd.Series([X])
        elif isinstance(X, list):
            X = pd.Series(X)
        elif isinstance(X, pd.DataFrame):
            if 'text' in X.columns:
                if 'title' in X.columns:
                    logging.info("Combining title and text columns...")
                    X = X['title'].astype(str) + " " + X['text'].astype(str)
                else:
                    X = X['text']
            elif X.shape[1] == 1:
                X = X.iloc[:, 0]
            else:
                logging.warning("No 'text' column found. Using the first column.")
                X = X.iloc[:, 0]

        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        logging.info("Starting text cleaning...")
        result = self._clean_series(X)
        logging.info("Text cleaning complete.")
        return result

    # ------------------------------------------------------------------ #
    #  Training-data helper                                               #
    # ------------------------------------------------------------------ #

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a raw training DataFrame.
        
        Combines 'title' + 'text' columns (if both present), applies full 
        text cleaning, and returns a copy of the DataFrame with a new 
        'cleaned_text' column ready for model training.
        
        Args:
            df (pd.DataFrame): Raw data with at least a 'text' column.
            
        Returns:
            pd.DataFrame: Copy of df with an added 'cleaned_text' column.
        """
        logging.info("Preprocessing training DataFrame...")

        # Ensure fitted
        if not self._final_stopwords and self.stopwords:
            self.fit()

        result_df = df.copy()

        # Combine title + text if both exist
        if 'title' in result_df.columns and 'text' in result_df.columns:
            combined = result_df['title'].astype(str) + " " + result_df['text'].astype(str)
        elif 'text' in result_df.columns:
            combined = result_df['text'].astype(str)
        else:
            raise ValueError("DataFrame must contain a 'text' column.")

        result_df['cleaned_text'] = self._clean_series(combined)

        # Drop rows where cleaned_text ended up empty
        before = len(result_df)
        result_df = result_df[result_df['cleaned_text'].str.len() > 0]
        dropped = before - len(result_df)
        if dropped:
            logging.info(f"Dropped {dropped} rows with empty text after cleaning.")

        logging.info(f"Preprocessing complete. {len(result_df)} rows remaining.")
        return result_df

    # ------------------------------------------------------------------ #
    #  URL detection and extraction                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_url(text: str) -> bool:
        """
        Check whether a string looks like a URL.
        
        Args:
            text (str): Input string.
            
        Returns:
            bool: True if the string is a URL.
        """
        text = text.strip()
        try:
            parsed = urlparse(text)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False

    def extract_text_from_url(self, url: str, timeout: int = 15) -> str:
        """
        Scrape article text from a URL by extracting <p> tag content.
        
        Args:
            url (str): The URL to scrape.
            timeout (int): Request timeout in seconds.
            
        Returns:
            str: Extracted article text.
            
        Raises:
            ValueError: If no text could be extracted.
            requests.RequestException: On network errors.
        """
        logging.info(f"Fetching article from URL: {url}")
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Extract text from <p> tags (most article content lives here)
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)

        # Fallback: if very little text from <p>, use full body text
        if len(text.split()) < 20:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)

        if not text.strip():
            raise ValueError(f"Could not extract any text from URL: {url}")

        logging.info(f"Extracted {len(text.split())} words from URL.")
        return text

    # ------------------------------------------------------------------ #
    #  Unified input handler (URL or raw text)                            #
    # ------------------------------------------------------------------ #

    def process_input(self, user_input: str) -> str:
        """
        Unified entry point for user input.
        
        Auto-detects whether the input is a URL or plain text.
        If URL → scrapes the article text first.
        In both cases → applies full text cleaning.
        
        Args:
            user_input (str): Raw text or a URL string.
            
        Returns:
            str: Cleaned text ready for model prediction.
        """
        # Ensure fitted
        if not self._final_stopwords and self.stopwords:
            self.fit()

        user_input = user_input.strip()

        if self.is_url(user_input):
            logging.info("Input detected as URL. Extracting article text...")
            raw_text = self.extract_text_from_url(user_input)
        else:
            logging.info("Input detected as plain text.")
            raw_text = user_input

        # Clean the text
        cleaned_series = self._clean_series(pd.Series([raw_text]))
        return cleaned_series.iloc[0]
