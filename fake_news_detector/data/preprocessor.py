
import re
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPreprocessor:
    """
    Class to preprocess text data for the Fake News Detector.
    """
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text.
        Args:
            text (str): Input text string.
        Returns:
            str: Cleaned text string.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers (optional, keeping basic punctuation might be useful for some models but TF-IDF usually handles it)
        # For this baseline, we'll keep it simple and just do basic cleaning.
        # Removing non-alphanumeric characters but keeping spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', title_column: str = 'title') -> pd.DataFrame:
        """
        Preprocesses the dataframe: combines title and text, cleans the result.
        Args:
            df (pd.DataFrame): Input dataframe.
            text_column (str): Name of the text column.
            title_column (str): Name of the title column.
        Returns:
            pd.DataFrame: Dataframe with a new 'cleaned_text' column.
        """
        logging.info("Preprocessing dataframe...")
        
        # Combine title and text if title column exists
        if title_column in df.columns:
            logging.info("Combining title and text...")
            df['full_text'] = df[title_column] + " " + df[text_column]
        else:
            df['full_text'] = df[text_column]
            
        logging.info("Cleaning text...")
        df['cleaned_text'] = df['full_text'].apply(self.clean_text)
        
        logging.info("Preprocessing complete.")
        return df
